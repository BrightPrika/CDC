def sample(model, x0, timesteps=1000):
    x = torch.randn_like(x0)  # Start from random noise
    for t in reversed(range(timesteps)):
        with torch.no_grad():
            noise_pred = model(x, torch.tensor([t]).float().to(device))
            x = noise_scheduler.step(x, noise_pred, t)
    return x  # Corrected image