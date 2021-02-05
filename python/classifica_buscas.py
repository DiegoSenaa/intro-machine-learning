import pandas as pd
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

training_percent = 0.9

training_length = int(training_percent * len(Y))
test_length = len(Y) - training_length

training_x_values = X[:training_length]
training_y_values = Y[:training_length]

test_x_values = X[-test_length:]
test_y_values = Y[-test_length:]

model = MultinomialNB()
model.fit(training_x_values, training_y_values)

result = model.predict(test_x_values)

result = model.predict(test_x_values)
differences = result - test_y_values

hits = [d for d in differences if d == 0]

total_of_hits = len(hits)
total_of_elements = len(test_x_values)

tax_of_hits = 100.0 * total_of_hits / total_of_elements

print('Taxa de acertos', tax_of_hits)
print('Total de elementos', total_of_elements)


