from ..models.LogRegression import LogRegression 

model = LogRegression()

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y = [0, 1, 1, 1]

model.add_data(X, y)

print("Initial weights:", model.weight)

# Test basic logistic regression (batch)
print("\nTraining with Batch Gradient Ascent...")
model.iterate_batch(epochs=200)
print("Weights after batch training:", model.weight)

print("Predictions after batch training:")
for x in X:
    print(f"{x} -> {model.predict(x)}")

# Reset model
model = LogRegression()
model.add_data(X, y)

# Test stochastic gradient ascent (SGA)
print("\nTraining with Stochastic Gradient Ascent...")
model.iterate_stochastic(epochs=200)
print("Weights after SGA training:", model.weight)

print("Predictions after SGA training:")
for x in X:
    print(f"{x} -> {model.predict(x)}")

# Reset model
model = LogRegression()
model.add_data(X, y)


# Test mini batch gradient ascent
print("\nTraining with Mini Batch Gradient Ascent...")
model.iterate_mini_batch(batch_size=5,epochs=200)
print("Weights after MBGA training:", model.weight)

print("Predictions after MBGA training:")
for x in X:
    print(f"{x} -> {model.predict(x)}")

# Reset model
model = LogRegression()
model.add_data(X, y)

# Test log reg momentum
print("\nTraining with Momentum Logistic Regressions...")
model.iterate_momentum(epochs=200)
print("Weights after Momentum training:", model.weight)

print("Predictions after momentum training:")
for x in X:
    print(f"{x} -> {model.predict(x)}")
