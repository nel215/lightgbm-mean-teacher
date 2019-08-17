import chainer
from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer.dataset import convert
from chainer.training.updaters import StandardUpdater


class Model(Chain):

    def __init__(self, in_size: int):
        super(Model, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(in_size, 64)
            self.l1 = L.Linear(None, 64)
            self.l2 = L.Linear(None, 1)

    def predict(self, x, raw=False):
        """
        Args:
            x: (bs, in_size)
        """
        h = F.dropout(self.embed(x))
        h = F.dropout(F.relu(self.l1(h)))
        h = self.l2(h)
        if not raw:
            h = F.sigmoid(h)
        return F.reshape(h, (-1,))

    def loss(self, pred_y, true_y):
        return F.sigmoid_cross_entropy(pred_y, true_y)

    def forward(self, x, y):
        """
        Returns:
            loss:
        """
        pred = self.predict(x, raw=True)
        loss = self.loss(pred, y)
        return loss


class MeanTeacherChain(Chain):

    def __init__(self, model):
        super(MeanTeacherChain, self).__init__()
        with self.init_scope():
            self.student = model
            self.teacher = model.copy('copy')

    def forward(self, train_x, train_y, test_x):
        # TODO: add noise
        student_train_pred = self.student.predict(train_x, raw=True)
        student_loss = self.student.loss(student_train_pred, train_y)
        student_test_pred = self.student.predict(test_x, raw=True)
        student_pred = F.concat(
            [student_train_pred, student_test_pred], axis=0)

        # teacher doesn't need gradient
        with chainer.using_config('enable_backprop', False):
            teacher_train_pred = self.teacher.predict(train_x, raw=True)
            teacher_loss = self.teacher.loss(teacher_train_pred, train_y)
            teacher_test_pred = self.teacher.predict(test_x, raw=True)
            teacher_pred = F.concat(
                [teacher_train_pred, teacher_test_pred], axis=0)

        # TODO: weight
        consistency_loss = F.mean_squared_error(teacher_pred, student_pred)
        loss = student_loss + consistency_loss
        chainer.report({
            'loss': loss,
            'teacher_loss': teacher_loss,
            'student_loss': student_loss,
            'consistency_loss': consistency_loss,
        }, self)
        return loss


class MeanTeacherUpdater(StandardUpdater):

    def __init__(self, ema_decay=0.99, *args, **kwargs):
        self.ema_decay = ema_decay
        super(MeanTeacherUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        train_iter = self._iterators['main']
        train_batch = train_iter.next()
        train_arrays = convert._call_converter(
            self.converter, train_batch, self.device)
        train_x, train_y = train_arrays
        test_batch = self._iterators['test'].next()
        test_x = convert._call_converter(
            self.converter, test_batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target
        optimizer.update(loss_func, train_x, train_y, test_x)

        # update teacher
        student = optimizer.target.student
        teacher = optimizer.target.teacher
        teacher_params = {k: v for k, v in teacher.namedparams()}
        for name, s in student.namedparams():
            t = teacher_params[name]
            t.data = self.ema_decay * t.data + (1.-self.ema_decay) * s.data

        if self.auto_new_epoch and train_iter.is_new_epoch:
            optimizer.new_epoch(auto=True)
