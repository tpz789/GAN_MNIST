from train import Trainer

HEIGHT = 28
WIDTH = 28
CHANNEL = 1
LATENT_SPACE_SIZE = 100
EPOCHS = 50001
BATCH = 32
CHECKPOINT = 500
# MODEL_TYPE = -1
MODEL_TYPE = 3

trainer = Trainer(height=HEIGHT, \
                  width=WIDTH, \
                  channels=CHANNEL, \
                  latent_size=LATENT_SPACE_SIZE, \
                  epochs=EPOCHS, \
                  batch=BATCH, \
                  checkpoint=CHECKPOINT,
                  model_type=MODEL_TYPE)

trainer.train()
trainer.plot()
