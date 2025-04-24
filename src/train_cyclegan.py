import torch
from Generator_Discriminator.py import Generator, Discriminator, CycleGANLoss

# Initialize models and move them to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)  # Under/Over-exposed → Normal
F = Generator().to(device)  # Normal → Under/Over-exposed
D_Y = Discriminator().to(device)  # Discriminator for normal
D_X = Discriminator().to(device)  # Discriminator for under/over

# Loss function
criterion = CycleGANLoss()

# Combine optimizers for G and F
opt_G_F = torch.optim.Adam(
    list(G.parameters()) + list(F.parameters()), lr=2e-4, betas=(0.5, 0.999)
)
opt_D_Y = torch.optim.Adam(D_Y.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D_X = torch.optim.Adam(D_X.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Number of epochs (make sure to define your dataloader elsewhere)
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get real images
        real_A = batch["under_over_exposed"].to(device)
        real_B = batch["normal_exposed"].to(device)

        # Generate fake images
        fake_B = G(real_A)
        fake_A = F(real_B)

        # Cycle consistency
        cycled_A = F(fake_B)
        cycled_B = G(fake_A)

        # Identity loss (optional)
        same_A = F(real_A)
        same_B = G(real_B)

        # --- Generator losses ---
        loss_G_adv = criterion.adversarial_loss(D_Y(fake_B), real=True)
        loss_F_adv = criterion.adversarial_loss(D_X(fake_A), real=True)

        loss_cycle = criterion.cycle_loss(real_A, cycled_A) + criterion.cycle_loss(real_B, cycled_B)
        loss_identity = criterion.identity_loss(real_A, same_A) + criterion.identity_loss(real_B, same_B)

        # Total generator loss
        total_loss_G_F = loss_G_adv + loss_F_adv + loss_cycle + loss_identity

        # Backprop for G and F
        opt_G_F.zero_grad()
        total_loss_G_F.backward()
        opt_G_F.step()

        # --- Discriminator Y (normal) ---
        loss_D_Y_real = criterion.adversarial_loss(D_Y(real_B), real=True)
        loss_D_Y_fake = criterion.adversarial_loss(D_Y(fake_B.detach()), real=False)
        loss_D_Y_total = loss_D_Y_real + loss_D_Y_fake

        opt_D_Y.zero_grad()
        loss_D_Y_total.backward()
        opt_D_Y.step()

        # --- Discriminator X (under/over) ---
        loss_D_X_real = criterion.adversarial_loss(D_X(real_A), real=True)
        loss_D_X_fake = criterion.adversarial_loss(D_X(fake_A.detach()), real=False)
        loss_D_X_total = loss_D_X_real + loss_D_X_fake

        opt_D_X.zero_grad()
        loss_D_X_total.backward()
        opt_D_X.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss_G_F: {total_loss_G_F.item():.4f}, "
          f"Loss_D_Y: {loss_D_Y_total.item():.4f}, Loss_D_X: {loss_D_X_total.item():.4f}")
