import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt


# инициализация весов для скрытых слоев и выходного слоя
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.weights_hidden = [np.random.randn(input_size, hidden_sizes[0])]
        for i in range(1, len(hidden_sizes)):
            self.weights_hidden.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]))

        self.weights_output = np.random.randn(hidden_sizes[-1], output_size)

# sigmoid и softmax реализуют функции активации
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# прямое распространение, проход через все слои с функциями активации
    def forward(self, x):
        hidden_output = x
        for weights in self.weights_hidden:
            hidden_input = np.dot(hidden_output, weights)
            hidden_output = self.sigmoid(hidden_input)

        output_input = np.dot(hidden_output, self.weights_output)
        output = self.softmax(output_input)
        return output
# backpropogation
    def train(self, x, y, learning_rate):
        hidden_output = x
        hidden_outputs = [x]
        for weights in self.weights_hidden:
            hidden_input = np.dot(hidden_output, weights)
            hidden_output = self.sigmoid(hidden_input)
            hidden_outputs.append(hidden_output)

        output_input = np.dot(hidden_output, self.weights_output)
        output = self.softmax(output_input)

        output_error = y - output
        output_delta = output_error * (output * (1 - output))

        hidden_errors = [output_delta.dot(self.weights_output.T)]
        hidden_deltas = [hidden_errors[-1] * (hidden_outputs[-1] * (1 - hidden_outputs[-1]))]

        for i in range(len(self.weights_hidden) - 1, 0, -1):
            hidden_errors.append(hidden_deltas[-1].dot(self.weights_hidden[i].T))
            hidden_deltas.append(hidden_errors[-1] * (hidden_outputs[i] * (1 - hidden_outputs[i])))

        hidden_deltas.reverse()

        self.weights_output += hidden_outputs[-1].T.dot(output_delta) * learning_rate
        for i in range(len(self.weights_hidden)):
            self.weights_hidden[i] += hidden_outputs[i].T.dot(hidden_deltas[i]) * learning_rate

    def save_weights(self, filename):
        np.savez(filename, weights_hidden=self.weights_hidden, weights_output=self.weights_output)

    def load_weights(self, filename):
        data = np.load(filename)
        self.weights_hidden = data['weights_hidden']
        self.weights_output = data['weights_output']

# графический интерфейс
class App:
    def __init__(self, master):
        self.master = master
        self.canvas_width = 256
        self.canvas_height = 256
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        self.recognize_button = tk.Button(master, text="Recognize", command=self.recognize)
        self.recognize_button.pack()
        self.show_weights_button = tk.Button(master, text="Show Weights", command=self.show_weights)
        self.show_weights_button.pack()
        self.neural_network = NeuralNetwork(16 * 16, [64, 64, 64], 10)
        self.load_weights()
        self.result_window = None
# обработка рисования
    def paint(self, event):
        d = 5
        x1, y1 = (event.x - d), (event.y - d)
        x2, y2 = (event.x + d), (event.y + d)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
# очищение холста
    def clear_canvas(self):
        self.canvas.delete("all")

# результат распознавания и исправление его
    def recognize(self):
        image = self.process_image()
        prediction = self.neural_network.forward(image.reshape(1, -1))
        self.result_window = tk.Toplevel(self.master)
        result_label = tk.Label(self.result_window, text=f"Prediction: {np.argmax(prediction)}")
        result_label.pack()
        correct_button = tk.Button(self.result_window, text="Correct", command=self.clear_canvas_and_close_windows)
        correct_button.pack()
        incorrect_button = tk.Button(self.result_window, text="Incorrect", command=self.train_with_correct_label)
        incorrect_button.pack()
# обработка изображения и преобразование его в массив 16х16
    def process_image(self):
        image = np.zeros((16, 16))
        chunk_width = self.canvas_width // 16
        chunk_height = self.canvas_height // 16
        for i in range(16):
            for j in range(16):
                x_start = i * chunk_width
                y_start = j * chunk_height
                chunk = []
                if self.canvas.find_overlapping(x_start, y_start, x_start + chunk_width, y_start + chunk_height):
                    chunk.append(1)
                else:
                    chunk.append(0)
                image[i][j] = np.mean(chunk)
        return image

# загрузка веса из файла
    def load_weights(self):
        try:
            self.neural_network.load_weights("weights.npz")
        except FileNotFoundError:
            print("No weights file found, initializing new weights...")

# сохранение текущего веса в файл
    def save_weights(self):
        self.neural_network.save_weights("weights.npz")

# очищение холста и закрытие окна с результатом
    def clear_canvas_and_close_windows(self):
        self.clear_canvas()
        self.result_window.destroy()
# обучение с правильной меткой и сохранение весов
    def train_with_correct_label(self):
        correct_label = simpledialog.askinteger("Correct Label", "Enter the correct label:")
        if correct_label is not None:
            image = self.process_image()
            target = np.zeros(10)
            target[correct_label] = 1
            self.neural_network.train(image.reshape(1, -1), target.reshape(1, -1), 0.01)
            messagebox.showinfo("Training", "Network has been trained with the correct label.")
            self.clear_canvas_and_close_windows()
            self.save_weights()
# отображение весов
    def show_weights(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        flattened_weights = []
        for layer_weights in self.neural_network.weights_hidden:
            flattened_weights.extend(layer_weights.flatten())
        flattened_weights.extend(self.neural_network.weights_output.flatten())

        # Normalize weights between 0 and 1
        flattened_weights = np.array(flattened_weights)
        normalized_weights = (flattened_weights - flattened_weights.min()) / (
                    flattened_weights.max() - flattened_weights.min())

        # Assign colors based on weight values
        colors = [(1 - w, w, 0) for w in normalized_weights]

        # Plot weights as colored squares
        ax.bar(range(len(flattened_weights)), np.ones_like(flattened_weights), color=colors)

        ax.set_title('Neural Network Weights')
        ax.set_xlabel('Weight Index')
        ax.set_ylabel('Value')

        plt.show()


root = tk.Tk()
app = App(root)
root.mainloop()
