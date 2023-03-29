import torch


classes = torch.arange(self.num_classes).long()
labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
mask = labels.eq(classes.expand(batch_size, self.num_classes))