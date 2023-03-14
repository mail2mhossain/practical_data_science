import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from autokeras import StructuredDataRegressor
from tensorflow.keras.callbacks import EarlyStopping

clean_data_df = pd.read_csv('clean_grouped_data.csv')
#print(clean_data_df.shape)
IP_ADDRESS_dummies = pd.get_dummies(clean_data_df['IP_ADDRESS'], drop_first=True)
clean_data_df = pd.concat([clean_data_df.drop('IP_ADDRESS', axis=1), IP_ADDRESS_dummies], axis=1)
PAIR_NAME_dummies = pd.get_dummies(clean_data_df['PAIR_NAME'], drop_first=True)
clean_data_df = pd.concat([clean_data_df.drop('PAIR_NAME', axis=1), PAIR_NAME_dummies], axis=1)
NAME_dummies = pd.get_dummies(clean_data_df['NAME'], drop_first=True)
clean_data_df = pd.concat([clean_data_df.drop('NAME', axis=1), NAME_dummies], axis=1)

print(clean_data_df.shape)

X = clean_data_df.drop('VEHICLES' , axis=1)
y = clean_data_df['VEHICLES']

test_size = int(X.shape[0] *.20)
print(test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)
#print (X_valid.shape, y_valid.shape)

# Initialize the AutoKeras regressor
regressor = StructuredDataRegressor(max_trials=15, loss='mean_squared_error', overwrite=True, metrics=['mse', 'mae'])

early_stop = EarlyStopping(patience=25) # monitor='val_loss', mode='min', verbose=1,

# Fit the regressor to the training data
regressor.fit(X_train, y_train, 
              validation_data=(X_valid, y_valid),
              epochs=600, verbose=0,  #batch_size=128, 
              callbacks=[early_stop])

# Evaluate the regressor on the test data
output_values = regressor.evaluate(X_test, y_test)
#print('Errors: ', error_values)
print('Mean Squared Error:', np.sqrt(output_values[1]))
print('Mean Absolute Error:', output_values[2])
#print('R2 Score:', r2)

# Make predictions with the regressor
predictions = regressor.predict(X_test).flatten()
real = y_test.to_numpy()
for i in range(10):
    print('Predicted price:', predictions[i].round(3))
    print('Real price:', real[i].round(0))
    print('')
# get the best performing model
model = regressor.export_model()
# summarize the loaded model
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model)

# Save the model
# model.save('evi_model.h5')

# Load the saved model
# model = StructuredDataRegressor.load('evi_model.h5', overwrite=True)
