# Losses
class CycleGANLoss:
    def __init__(self, lambda_cycle=10.0, lambda_identity=5.0):
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def adversarial_loss(self, real, fake):
        return self.mse_loss(fake, torch.ones_like(fake))

    def cycle_loss(self, real, cycled):
        return self.l1_loss(real, cycled) * self.lambda_cycle

    def identity_loss(self, real, same):
        return self.l1_loss(real, same) * self.lambda_identity