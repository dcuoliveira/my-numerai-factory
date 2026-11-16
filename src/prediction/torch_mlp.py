"""
A torch MLP that behaves like sklearn's MLPRegressor, to the extent it can.

    from src.prediction.torch_mlp import TorchMLPRegressor

This exists to answer one question: how much of the sklearn MLP's behaviour on
this panel is the ARCHITECTURE, and how much is sklearn's particular training
loop? Answering it needs an implementation that differs in the framework and in
nothing else, so every documented detail of sklearn's loop is reproduced here
rather than replaced with a torch idiom.

It lives in its own module, not in prediction/models.py, because that module is
imported by run_forecast and by src/live/refit.py and must stay importable with
no torch installed (torch is OPTIONAL -- see setup.sh). TorchMLPWrapper imports
this file inside __init__, the same lazy pattern LGBWrapper uses for lightgbm.

------------------------------------------------------------------------------
What is matched, and where each rule came from
------------------------------------------------------------------------------

Every line below was read out of the installed sklearn, not recalled:

  loss        0.5 * mean((y - yhat)^2)                  _base.squared_loss
  L2          + 0.5 * alpha * sum(W^2) / batch_size     _multilayer_perceptron:361
              -- on the WEIGHTS only, never the biases, and divided by the size
              of the batch the term is computed on, not the training set
  init        uniform(-b, b), b = sqrt(6 / (fan_in + fan_out))    _init_coef:445
              -- Glorot, factor 6.0 for relu
  optimiser   Adam(lr=learning_rate_init, betas=(0.9, 0.999), eps=1e-8)
  batching    shuffle the INDICES each epoch, then contiguous slices; the last
              batch is the short remainder, not dropped        gen_batches
  stopping    epoch loss > best_loss - tol  ->  no_improvement += 1, else 0;
              stop once no_improvement > n_iter_no_change      :759, :823
  weights     with early_stopping=False sklearn keeps the FINAL weights and
              does NOT restore the best epoch's                :794
              -- so this does not restore them either, however tempting

Two differences remain and neither can be removed:

  * FLOAT ORDER. sklearn accumulates in float64 with numpy's BLAS; torch here
    also uses float64 but its own kernels, so the two diverge slowly. This is a
    replication, not a bit-for-bit port.
  * RNG. sklearn draws its init from numpy's Generator, torch from its own.
    Seeding both makes each reproducible; it does not make them equal. The
    right check is therefore agreement in DISTRIBUTION over seeds, not equality
    on one.

------------------------------------------------------------------------------
Why it is an sklearn estimator rather than a training script
------------------------------------------------------------------------------

prediction/models.py's contract is that .ModelClass is an INSTANCE which
hyper_params_search() hands to sklearn.base.clone and RandomizedSearchCV. clone
rebuilds an estimator from get_params(), so every constructor argument is stored
unmodified on self under its own name and nothing is derived until fit(). That
is the whole reason this subclasses BaseEstimator instead of wrapping a bespoke
loop: it buys the walk-forward, the purged CV, the scorers and the results tree
for free.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class IdPassthroughScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler for the real features; the id columns pass through, last.

    An integer ticker code is not a quantity. Standardising it produces a number
    with no meaning at all, and -- worse -- one an embedding lookup cannot use,
    since the lookup needs the original integer. So the id columns are lifted
    out, everything else is scaled, and the ids are appended UNCHANGED at the
    end.

    Putting them last is what lets the estimator find them without knowing column
    names: sklearn transformers hand numpy downstream, so a name-based contract
    would break the moment anything else sat between here and the network.
    `n_embedding_columns` on the estimator is simply how many trailing columns
    are ids, and this class is what makes that true.

    Columns are matched by NAME when X is a DataFrame -- which it is, all the way
    from walk_forward_fit_and_predict -- and the class degrades to scaling
    everything when it is handed a bare array with no names to match.

    :param id_columns: column names to pass through unscaled
    :type id_columns: tuple of str
    """

    def __init__(self, id_columns=()):
        self.id_columns = id_columns

    def fit(self, X, y=None):
        from sklearn.preprocessing import StandardScaler

        names = list(getattr(X, "columns", []))
        wanted = [c for c in self.id_columns if c in names]
        self.id_columns_ = wanted
        self.feature_columns_ = [c for c in names if c not in set(wanted)]

        frame = X[self.feature_columns_] if names else X
        self.scaler_ = StandardScaler().fit(np.asarray(frame, dtype="float64"))
        self.n_ids_ = len(wanted)
        return self

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        if self.id_columns_:
            scaled = self.scaler_.transform(
                np.asarray(X[self.feature_columns_], dtype="float64"))
            ids = np.asarray(X[self.id_columns_], dtype="float64")
            return np.hstack([scaled, ids])
        return self.scaler_.transform(np.asarray(X, dtype="float64"))


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """
    Feedforward regressor on torch, matching sklearn's MLPRegressor loop.

    :param hidden_layer_sizes: widths of the hidden layers
    :type hidden_layer_sizes: tuple of int
    :param activation: "relu" or "identity". identity inserts NO nonlinearity
                       between the hidden layers, so the whole stack composes to a
                       single affine map -- matching sklearn
                       MLPRegressor(activation='identity'). Named so the parameter
                       grid stays interchangeable with sklearn's.
    :type activation: str
    :param alpha: L2 penalty, applied exactly as sklearn applies it
    :type alpha: float
    :param batch_size: rows per optimiser step; clipped to len(X) like sklearn's
    :type batch_size: int
    :param learning_rate_init: Adam's step size
    :type learning_rate_init: float
    :param max_iter: epoch ceiling; the stopping rule usually bites first
    :type max_iter: int
    :param shuffle: reshuffle the sample order each epoch
    :type shuffle: bool
    :param random_state: seeds init and shuffling
    :type random_state: int or None
    :param tol: improvement below this does not count as improvement
    :type tol: float
    :param n_iter_no_change: stop after this many non-improving epochs
    :type n_iter_no_change: int
    :param device: "cpu" or "cuda"; cpu by default because the walk-forward runs
                   many small fits and the transfers cost more than they save
    :type device: str
    :param n_embedding_columns: how many TRAILING columns of X are integer
                                category codes rather than features.
                                IdPassthroughScaler is what puts them there. 0
                                disables the embedding entirely and the network
                                is then byte-identical to the version that had no
                                embedding code at all.
    :type n_embedding_columns: int
    :param embed_dim: width of each learned category vector
    :type embed_dim: int

    --------------------------------------------------------------------------
    The embedding, and why it is built the way LSTMForecastNetwork builds one
    --------------------------------------------------------------------------

    prediction/nn/model.py already solved this for the sequence model, and the
    two should not disagree about what a symbol embedding is. Copied from it:

      * nn.Embedding(n_train_symbols + 1, embed_dim, padding_idx=n_train_symbols).
        The extra row absorbs codes absent from the TRAINING block. padding_idx
        zero-initialises it and holds its gradient at zero, so an unseen ticker
        contributes no identity signal rather than a random one -- which is what
        an untrained random row would do.
      * The index is built from the training block only. A code first seen at
        predict time maps to the padding row; it does not get a vector, and it
        must not, because there is nothing to have learned one from.
      * The vector is CONCATENATED to the features, not added, so the first
        hidden layer sees [scaled features | ticker vector] and can form
        interactions between the two.

    What differs is only what follows: the LSTM feeds the concatenation to a
    recurrent layer, this feeds it to the same dense stack the no-embedding
    variant uses. That is the one thing varying between the two arms.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                 batch_size=1024, learning_rate_init=1e-3, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4, n_iter_no_change=10,
                 device="cpu", n_embedding_columns=0, embed_dim=8,
                 loss="mse", loss_kwargs=None,
                 n_period_columns=0, n_reference_columns=0):
        # Stored verbatim, nothing derived: clone() rebuilds from get_params()
        # and sklearn's check_estimator forbids mutating arguments here.
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.device = device
        self.n_embedding_columns = n_embedding_columns
        self.embed_dim = embed_dim
        # Training objective. "mse" keeps the byte-identical control loop below;
        # every other name routes through torch_mlp_losses.build_torch_mlp_loss.
        # loss_kwargs carries per-loss options (bmc guards, sharpe/calmar eps/beta).
        # n_period_columns / n_reference_columns count TRAILING passthrough columns
        # the loss needs (an integer period code, a reference series), carried the
        # same way n_embedding_columns carries the ticker id -- never fed to the net.
        self.loss = loss
        self.loss_kwargs = loss_kwargs
        self.n_period_columns = n_period_columns
        self.n_reference_columns = n_reference_columns

    # ----------------------------------------------------------------------- #

    def _split_ids(self, X):
        """
        Peel the trailing passthrough columns off X.

        Layout, when present, is [features | embed id | period | reference],
        appended UNSCALED and in that order by IdPassthroughScaler. They are
        routed to the embedding (id) and the loss (period, reference); never to
        the network input.

        :return: (features, embed_codes|None, period_codes|None, reference|None)
        :rtype: (np.ndarray, np.ndarray|None, np.ndarray|None, np.ndarray|None)
        """
        n_extra = self._n_embed + self._n_period + self._n_reference
        if not n_extra:
            return X, None, None, None
        features = X[:, :-n_extra]
        off = X.shape[1] - n_extra
        codes = period = reference = None
        if self._n_embed:
            codes = X[:, off].astype("int64"); off += self._n_embed
        if self._n_period:
            period = X[:, off].astype("int64"); off += self._n_period
        if self._n_reference:
            reference = X[:, off].astype("float64"); off += self._n_reference
        return features, codes, period, reference

    def _index_codes(self, codes):
        """
        Map raw codes to embedding rows, reserving the last row for unseen ones.

        Built from the TRAINING codes only. A code first met at predict time has
        nothing behind it to have learned a vector from, so it goes to the
        padding row rather than colliding with some other ticker's.

        :param codes: raw integer codes
        :return: row indices into the embedding table
        :rtype: np.ndarray
        """
        rows = np.full(len(codes), self._padding_row, dtype="int64")
        for position, code in enumerate(codes):
            row = self._code_to_row.get(int(code))
            if row is not None:
                rows[position] = row
        return rows

    def _build(self, n_features, torch, nn, generator):
        """
        Assemble the stack and initialise it the way sklearn does.

        torch's own nn.Linear default is Kaiming-uniform over fan_in alone, which
        is a different distribution from Glorot over fan_in + fan_out. Left alone
        it would be a second uncontrolled difference between the two
        implementations, so the weights are overwritten here.

        :return: the network, on the requested device, in float64
        :rtype: torch.nn.Sequential
        """
        sizes = self.hidden_layer_sizes
        if isinstance(sizes, (int, np.integer)):
            sizes = (int(sizes),)
        # The embedding is concatenated, so it widens the first layer's input.
        first = n_features + (int(self.embed_dim) if self._n_embed else 0)
        widths = [first] + [int(w) for w in sizes] + [1]

        # relu between the hidden layers; identity means NONE, so the Linear
        # stack composes to one affine map -- exactly sklearn's
        # activation='identity'. The head is always linear either way. The Glorot
        # bound below uses factor 6.0, which is sklearn's for both relu and
        # identity (only 'logistic' takes 2.0), so init matches too.
        hidden_activation = nn.ReLU if self.activation == "relu" else None
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2 and hidden_activation is not None:
                layers.append(hidden_activation())            # none on the head
        net = nn.Sequential(*layers).to(dtype=torch.float64, device=self.device)

        with torch.no_grad():
            for layer in net:
                if not isinstance(layer, nn.Linear):
                    continue
                fan_in, fan_out = layer.in_features, layer.out_features
                bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
                # sklearn draws BOTH the weights and the bias from this bound.
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)
        return net

    def fit(self, X, y):
        """
        Train, stopping on the training loss exactly as sklearn does.

        :param X: (n_samples, n_features)
        :param y: (n_samples,)
        :return: self
        """
        import torch
        from torch import nn

        if self.activation not in ("relu", "identity"):
            raise ValueError(
                "only activation in {{'relu', 'identity'}} is implemented, got "
                "{!r}. identity gives a linear stack matching sklearn's "
                "activation='identity'; relu is the nonlinear default. The "
                "parameter stays interchangeable with sklearn's grid.".format(
                    self.activation)
            )

        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64").ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D, got shape {}".format(X.shape))
        if len(X) != len(y):
            raise ValueError("X has {} rows, y has {}".format(len(X), len(y)))

        self.n_features_in_ = X.shape[1]
        self._n_embed = int(self.n_embedding_columns)
        self._n_period = int(self.n_period_columns)
        self._n_reference = int(self.n_reference_columns)
        n_extra = self._n_embed + self._n_period + self._n_reference
        if min(self._n_embed, self._n_period, self._n_reference) < 0 or n_extra >= X.shape[1]:
            raise ValueError(
                "trailing passthrough columns (embed={}, period={}, reference={}) "
                "are impossible for X with {} columns; at least one real feature "
                "must remain".format(self._n_embed, self._n_period,
                                     self._n_reference, X.shape[1])
            )
        if self._n_embed > 1:
            raise NotImplementedError(
                "one embedded column is supported; {} were asked for. A second "
                "category would need its own table and its own width, and no "
                "study has asked for one.".format(self._n_embed)
            )

        X, codes, period_codes, reference = self._split_ids(X)

        # Build the training objective. "mse" stays inline below (byte-identical
        # control); every other loss is a callable fn(signal, target, reference,
        # period) from torch_mlp_losses, with meta telling us whether it needs a
        # single full-window step per epoch and which passthrough columns it uses.
        from src.prediction.torch_mlp_losses import build_torch_mlp_loss
        loss_fn, loss_meta = build_torch_mlp_loss(
            self.loss, **(self.loss_kwargs or {}))
        if loss_meta["needs_period"] and period_codes is None:
            raise ValueError(
                "loss=%r needs a per-period grouping but no period column was "
                "plumbed (n_period_columns=0)" % (self.loss,))
        if loss_meta["needs_reference"] and reference is None:
            raise ValueError(
                "loss=%r needs a reference series but no reference column was "
                "plumbed (n_reference_columns=0)" % (self.loss,))

        device = torch.device(self.device)

        seed = 0 if self.random_state is None else int(self.random_state)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shuffler = np.random.RandomState(seed)

        embedding = None
        if self._n_embed:
            # One row per code SEEN IN TRAINING, plus one padding row at the end
            # for everything else. Exactly LSTMForecastNetwork's construction.
            distinct = sorted(set(int(c) for c in codes))
            self._code_to_row = {code: row for row, code in enumerate(distinct)}
            self._padding_row = len(distinct)
            embedding = nn.Embedding(
                len(distinct) + 1, int(self.embed_dim),
                padding_idx=self._padding_row,
            ).to(dtype=torch.float64, device=device)
            with torch.no_grad():
                # Match the dense layers' Glorot bound rather than torch's N(0,1)
                # default, so the embedding does not enter the first layer an
                # order of magnitude larger than the features beside it.
                bound = float(np.sqrt(6.0 / (len(distinct) + 1 + int(self.embed_dim))))
                embedding.weight.uniform_(-bound, bound, generator=generator)
                embedding.weight[self._padding_row].zero_()   # padding_idx stays 0
            self.n_symbols_ = len(distinct)

        net = self._build(X.shape[1], torch, nn, generator)
        if str(device) != "cpu":
            net = net.to(device)
        parameters = list(net.parameters())
        if embedding is not None:
            parameters += list(embedding.parameters())
        optimiser = torch.optim.Adam(
            parameters, lr=float(self.learning_rate_init),
            betas=(0.9, 0.999), eps=1e-8,
        )

        Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
        yt = torch.as_tensor(y, dtype=torch.float64, device=device).reshape(-1, 1)
        rows_t = None
        if self._n_embed:
            rows_t = torch.as_tensor(
                self._index_codes(codes), dtype=torch.long, device=device)
        period_t = (torch.as_tensor(period_codes, dtype=torch.long, device=device)
                    if period_codes is not None else None)
        reference_t = (torch.as_tensor(reference, dtype=torch.float64, device=device)
                       if reference is not None else None)

        # sklearn clips a batch larger than the sample and warns; the clip is the
        # part that matters, and on a purged fold this fires often -- the smallest
        # inner fold on this panel is ~1,800 rows against a 8,192 grid entry.
        batch = int(max(1, min(int(self.batch_size), len(X))))
        weights = [m.weight for m in net if isinstance(m, nn.Linear)]

        all_rows = torch.arange(len(X), dtype=torch.long, device=device)

        def _forward(rows):
            batch_x = Xt[rows]
            if embedding is not None:
                batch_x = torch.cat([batch_x, embedding(rows_t[rows])], dim=1)
            return net(batch_x)

        def _data_loss(out, rows):
            # The MSE control stays inline and byte-identical -- do NOT route it
            # through the loss module. 0.5 * mean squared error = sklearn's
            # squared_loss, not MSE.
            if self.loss == "mse":
                return 0.5 * ((yt[rows] - out) ** 2).mean()
            ref = reference_t[rows] if reference_t is not None else None
            per = period_t[rows] if period_t is not None else None
            return loss_fn(out, yt[rows], reference=ref, period=per)

        def _l2(rows):
            # L2 on the WEIGHTS only, scaled by this batch's size. Reproducing the
            # /batch scaling matters: alpha's effective strength otherwise moves
            # with the batch size, and batch_size is in the search grid. Unchanged
            # from the MSE control, so alpha means the same in every loss arm.
            penalty = sum((w * w).sum() for w in weights)
            if embedding is not None:
                # Regularised alongside the dense weights so the embedding arm
                # does not differ in TWO things.
                penalty = penalty + (embedding.weight ** 2).sum()
            return (0.5 * float(self.alpha)) * penalty / len(rows)

        best_loss = np.inf
        no_improvement = 0
        self.loss_curve_ = []
        self.n_iter_ = 0
        self.diverged_ = False
        full_batch = loss_meta["full_batch"]

        for _ in range(int(self.max_iter)):
            if full_batch:
                # sharpe/calmar are ratios over the WHOLE per-period distribution,
                # meaningless in a minibatch spanning ~2 dates -- one gradient step
                # per epoch over every row.
                optimiser.zero_grad(set_to_none=True)
                out = _forward(all_rows)
                loss = _data_loss(out, all_rows) + _l2(all_rows)
                if not torch.isfinite(loss):
                    self.diverged_ = True
                    break
                loss.backward()
                optimiser.step()
                epoch_loss = float(loss.detach())
            else:
                order = np.arange(len(X))
                if self.shuffle:
                    order = shuffler.permutation(order)
                index = torch.as_tensor(order, dtype=torch.long, device=device)
                total, seen = 0.0, 0
                for start in range(0, len(X), batch):
                    rows = index[start:start + batch]
                    optimiser.zero_grad(set_to_none=True)
                    out = _forward(rows)
                    loss = _data_loss(out, rows) + _l2(rows)
                    if not torch.isfinite(loss):
                        # A diverged batch would poison every parameter on the
                        # next step; skip it and flag the run rather than propagate
                        # NaN. (Never fires for the MSE control -> byte-identical.)
                        self.diverged_ = True
                        continue
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * len(rows)
                    seen += len(rows)
                epoch_loss = total / max(seen, 1)

            self.loss_curve_.append(epoch_loss)
            self.n_iter_ += 1

            # sklearn's rule, transcribed: a non-improvement is measured against
            # best - tol, and the counter must EXCEED n_iter_no_change to stop.
            if epoch_loss > best_loss - float(self.tol):
                no_improvement += 1
            else:
                no_improvement = 0
            if epoch_loss < best_loss:
                best_loss = epoch_loss
            if no_improvement > int(self.n_iter_no_change):
                break

        # Deliberately NOT restoring the best epoch: with early_stopping=False
        # sklearn keeps the final weights (_multilayer_perceptron.py:794), and a
        # replication that quietly improved on its target would not be one.
        self.best_loss_ = best_loss
        # Training-dynamics diagnostics (persisted by the E42/E43 report, ignored
        # by a normal run). final vs best shows late instability; pred_std_ is the
        # collapse proxy -- a constant prediction (std ~ 0) is what an
        # over-regularised or diverged net produces.
        self.final_loss_ = self.loss_curve_[-1] if self.loss_curve_ else float("nan")
        with torch.no_grad():
            self.pred_std_ = float(_forward(all_rows).std().detach().cpu())
        self._net = net
        self._embedding = embedding
        self._torch = torch
        return self

    def predict(self, X):
        """
        :param X: (n_samples, n_features)
        :return: (n_samples,) predictions
        :rtype: np.ndarray
        """
        check_is_fitted(self, "_net")
        torch = self._torch
        device = torch.device(self.device)
        X = np.asarray(X, dtype="float64")
        # period / reference (if plumbed) are peeled and ignored at predict time
        # -- they exist only to shape the training loss.
        X, codes, _period, _reference = self._split_ids(X)
        self._net.eval()
        with torch.no_grad():
            batch_x = torch.as_tensor(X, dtype=torch.float64, device=device)
            if self._embedding is not None:
                self._embedding.eval()
                rows = torch.as_tensor(
                    self._index_codes(codes), dtype=torch.long, device=device)
                batch_x = torch.cat([batch_x, self._embedding(rows)], dim=1)
            out = self._net(batch_x)
        return out.cpu().numpy().ravel()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
