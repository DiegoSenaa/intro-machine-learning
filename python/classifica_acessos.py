from leito_csv import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

training_x_values = X[:90]
training_y_values = Y[:90]

test_x_values = X[-9:]
test_y_values = Y[-9:]

model = MultinomialNB()
model.fit(training_x_values, training_y_values)

result = model.predict(test_x_values)
differences = result - test_y_values

hits = [d for d in differences if d == 0]

total_of_hits = len(hits)
total_of_elements = len(test_x_values)

tax_of_hits = 100.0 * total_of_hits / total_of_elements

print('Taxa de acertos', tax_of_hits)
print('Total de elementos', total_of_elements)
