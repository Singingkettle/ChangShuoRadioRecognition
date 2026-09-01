import unittest

import torch

from csrr.evaluation.metrics import (Accuracy, Loss, StreamingAccuracy,
                                     StreamingLoss)


def samples(scores, labels, losses):
    return [{
        'pred_score': score,
        'gt_label': label.reshape(1),
        'classification_loss': loss.reshape(1),
    } for score, label, loss in zip(scores, labels, losses)]


class TestStreamingMetrics(unittest.TestCase):

    def test_matches_collecting_metrics(self):
        generator = torch.Generator().manual_seed(11)
        scores = torch.randn(37, 9, generator=generator).softmax(1)
        labels = torch.randint(0, 9, (37,), generator=generator)
        losses = torch.rand(37, generator=generator)
        records = samples(scores, labels, losses)
        original_accuracy = Accuracy(topk=(1,))
        original_loss = Loss(task='classification')
        streaming_accuracy = StreamingAccuracy(37, 1, topk=(1,))
        streaming_loss = StreamingLoss(37, 1, task='classification')
        for start in range(0, len(records), 8):
            batch = records[start:start + 8]
            original_accuracy.process(None, batch)
            original_loss.process(None, batch)
            streaming_accuracy.process(None, batch)
            streaming_loss.process(None, batch)
        original = {**original_accuracy.evaluate(37),
                    **original_loss.evaluate(37)}
        streaming = {**streaming_accuracy.evaluate(37),
                     **streaming_loss.evaluate(37)}
        self.assertEqual(original['accuracy/top1'],
                         streaming['accuracy/top1'])
        self.assertAlmostEqual(original['loss/classification'],
                               streaming['loss/classification'], places=7)

    def test_rejects_padded_or_wrong_size_protocol(self):
        with self.assertRaises(ValueError):
            StreamingAccuracy(5, 2)
        metric = StreamingAccuracy(4, 1)
        metric.process(None, samples(
            torch.eye(2), torch.tensor([0, 1]), torch.zeros(2)))
        with self.assertRaises(RuntimeError):
            metric.evaluate(2)


if __name__ == '__main__':
    unittest.main()
