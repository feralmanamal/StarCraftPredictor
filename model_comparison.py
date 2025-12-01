


residuals = y_train - predictions
plt.figure(figsize=(10, 6))

# Plot the residuals (errors) against the predicted values
# A good model will have these points randomly scattered around the zero line.
plt.scatter(predictions, residuals, alpha=0.5)

# Add the zero reference line
plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), color='red', linestyle='--')

plt.title('Residual Analysis: Errors vs. Predicted League Index')
plt.xlabel('Predicted League Index (P)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()




hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure(figsize=(12, 5))

# mse plot
plt.subplot(1, 2, 1)
plt.plot(hist['epoch'], hist['loss'], label='Training Loss (MSE)')
plt.plot(hist['epoch'], hist['val_loss'], label='Validation Loss (MSE)')
plt.title('Training and Validation Loss (MSE) Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# mae plot
plt.subplot(1, 2, 2)
plt.plot(hist['epoch'], hist['mae'], label='Training MAE')
plt.plot(hist['epoch'], hist['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE Over Time')
plt.xlabel('Epoch')
plt.ylabel('MAE (League Index Error)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()