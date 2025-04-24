# Training loop
model = DiffusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        x0 = batch["under_over_exposed"].to(device)  # Input image
        y0 = batch["normal_exposed"].to(device)     # Target image
        
        # Random timestep
        t = torch.randint(0, timesteps, (x0.shape[0], 1)).float().to(device)
        
        # Noise addition
        noise = torch.randn_like(y0)
        noisy_image = noise_scheduler.add_noise(y0, noise, t)
        
        # Predict noise
        predicted_noise = model(noisy_image, t)
        
        # Loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()