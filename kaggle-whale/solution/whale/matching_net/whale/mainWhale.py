from data_loader import WhaleNShotDataset
from WhaleBuilder import WhaleBuilder
import tqdm

# Experiment setup
batch_size = 2
fce = False

classes_per_set = 20
samples_per_class = 1
num_channels = 1
# Training setup
total_epochs = 100
total_train_batches = 1000
best_acc = 0.0
keep_prob = 0.1
image_size= 384


data = WhaleNShotDataset(batch_size=batch_size, classes_per_set=classes_per_set,
                            samples_per_class=samples_per_class, seed=2018, shuffle=True, use_cache=False)
obj_oneShotBuilder = WhaleBuilder(data)
obj_oneShotBuilder.build_experiment(batch_size=batch_size, num_channels=num_channels, lr=1e-3, image_size=image_size, classes_per_set=classes_per_set,
                                    samples_per_class=samples_per_class, keep_prob=0.1, fce=fce, optim="adam", weight_decay=0,
                                    use_cuda=True)

with tqdm.tqdm(total=total_train_batches) as pbar_e:
    for e in range(total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches)
        print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
        # total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_val_epoch(total_val_batches)
        # print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
        if total_accuracy > best_acc:
            best_acc = total_accuracy
            # total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_test_epoch(total_test_batches)
            # print("Epoch {}: test_loss:{} test_accuracy:{}".format(e, total_test_c_loss, total_test_accuracy))
        pbar_e.update(1)

