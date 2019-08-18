from chainer import reporter as reporter_module
from chainer.dataset import convert
from chainer.training.extensions import Evaluator
from chainer.backends import cuda
from sklearn.metrics import roc_auc_score


class AUCEvaluator(Evaluator):

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        iterator.reset()
        it = iterator

        summary = reporter_module.DictSummary()

        true_y = []
        pred_y = []
        for batch in it:
            in_arrays = convert._call_converter(
                self.converter, batch, self.device)
            assert isinstance(in_arrays, tuple)
            x, y = in_arrays
            true_y.append(y)
            pred_y.append(target.predict(x).data)
        auc = roc_auc_score(
            cuda.to_cpu(target.xp.concatenate(true_y, axis=0)),
            cuda.to_cpu(target.xp.concatenate(pred_y, axis=0)),
        )

        summary.add({f'{self.name}/main/auc': auc})

        return summary.compute_mean()
