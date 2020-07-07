class MelGANparams():
    def __init__(self,
                 discriminator_optimizer_lr=1e-4,
                 generator_optimizer_lr=1e-4,

                 ):

        self.discriminator_optimizer_lr = discriminator_optimizer_lr
        self.generator_optimizer_lr = generator_optimizer_lr
