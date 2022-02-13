import dataloader

DATA_DIR = './data/'

train_loader, validation_loader, test_loader = dataloader.load_data(DATA_DIR+'NoisySphericalPendulum.npy', batch_size=1, nb_workers=0)