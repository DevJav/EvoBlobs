import random
import math
import matplotlib.pyplot as plt
import networkx as nx
from constants import *
import copy

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = 0.0
        self.activation_function = relu

class Brain:
    def __init__(self):
        self.inputs: list[Input] = []
        self.outputs: list[Output] = []
        self.hidden_layers: list[list[Neuron]] = []
        self.hidden_layer: list[Neuron] = []
        self.output_layer: list[Neuron] = []

    def generate_brain(self):
        number_of_inputs = len(self.inputs)
        number_of_outputs = len(self.outputs)
        number_of_hidden_layers = random.randint(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS)

        self.hidden_layers = []

        neurons_first = random.randint(MIN_NEURONS_PER_LAYER, MAX_NEURONS_PER_LAYER)
        first_hidden = [Neuron(number_of_inputs) for _ in range(neurons_first)]
        self.hidden_layers.append(first_hidden)
        previous_neurons = neurons_first

        for _ in range(1, number_of_hidden_layers):
            neurons_this_layer = random.randint(MIN_NEURONS_PER_LAYER, MAX_NEURONS_PER_LAYER)
            hidden = [Neuron(previous_neurons) for _ in range(neurons_this_layer)]
            self.hidden_layers.append(hidden)
            previous_neurons = neurons_this_layer

        self.output_layer = [Neuron(previous_neurons) for _ in range(number_of_outputs)]

        self.plot_brain()

    def plot_brain(self, input_vals=None, hidden_vals=None, output_vals=None, manual=False):
        if not PLOT_BRAIN and not manual:
            return
        fig = plt.figure(num="Brain")
        fig.clf()

        G = nx.DiGraph()
        labels = {}
        node_colors = []
        edge_colors = []
        edge_widths = []

        # Input layer
        for i, inp in enumerate(self.inputs):
            name = f"{inp.name}"
            value = input_vals[i] if input_vals else 0
            G.add_node(name, pos=(0, i), layer='input')
            labels[name] = f"{name}\n{value:.2f}"
            node_colors.append('lightgreen')

        layer_offset = 1
        all_hidden_neurons = []
        prev_layer_names = [f"{inp.name}" for inp in self.inputs]

        # Hidden layers
        for l_idx, layer in enumerate(self.hidden_layers):
            layer_names = []
            for n_idx, neuron in enumerate(layer):
                name = f"H{l_idx}_{n_idx}"
                all_hidden_neurons.append(neuron)
                value = hidden_vals[len(all_hidden_neurons) - 1] if hidden_vals else 0
                G.add_node(name, pos=(layer_offset, n_idx), layer='hidden')
                labels[name] = f"{name}\n{value:.2f}"
                node_colors.append('lightblue')
                layer_names.append(name)

                # Connect previous layer
                for j, prev_name in enumerate(prev_layer_names):
                    weight = neuron.weights[j]
                    G.add_edge(prev_name, name, weight=weight)
                    edge_colors.append('green' if weight > 0 else 'red')
                    edge_widths.append(abs(weight) * 2)

            prev_layer_names = layer_names
            layer_offset += 1

        # Output layer
        for i, neuron in enumerate(self.output_layer):
            name = f"{self.outputs[i].name}"
            value = output_vals[i] if output_vals else 0
            G.add_node(name, pos=(layer_offset, i), layer='output')
            labels[name] = f"{name}\n{value:.2f}"
            node_colors.append('lightcoral')

            for j, prev_name in enumerate(prev_layer_names):
                weight = neuron.weights[j]
                G.add_edge(prev_name, name, weight=weight)
                edge_colors.append('green' if weight > 0 else 'red')
                edge_widths.append(abs(weight) * 2)

        # Highlight activated output
        if output_vals:
            max_output = max(output_vals)
            activated_neuron_index = next(i for i, neuron in enumerate(self.output_layer) if neuron.output == max_output)
            output_name = f"{self.outputs[activated_neuron_index].name}"
            labels[output_name] += "\nActivated"
            output_pos = layer_offset
            node_colors[-len(self.output_layer) + activated_neuron_index] = 'yellow'

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, labels=labels, node_color=node_colors, node_size=2000,
                edge_color=edge_colors, width=edge_widths, font_size=8, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos,
                                    edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)},
                                    font_size=7)

    def forward(self):
        for inp in self.inputs:
            inp.value = (inp.get() - inp.min_value) / (inp.max_value - inp.min_value)

        prev_outputs = [inp.value for inp in self.inputs]

        for layer in self.hidden_layers:
            current_outputs = []
            for neuron in layer:
                neuron.output = sum(w * val for w, val in zip(neuron.weights, prev_outputs)) + neuron.bias
                neuron.output = neuron.activation_function(neuron.output)
                current_outputs.append(neuron.output)
            prev_outputs = current_outputs

        # for neuron in self.output_layer:
        #     neuron.output = sum(w * val for w, val in zip(neuron.weights, prev_outputs)) + neuron.bias
        #     neuron.output = neuron.activation_function(neuron.output)
        raw_outputs = []
        for neuron in self.output_layer:
            neuron.output = sum(w * val for w, val in zip(neuron.weights, prev_outputs)) + neuron.bias
            raw_outputs.append(neuron.output)

        # Apply softmax to raw outputs
        exp_outputs = [math.exp(o) for o in raw_outputs]
        sum_exp = sum(exp_outputs)
        softmax_outputs = [eo / sum_exp for eo in exp_outputs]

        for neuron, soft_out in zip(self.output_layer, softmax_outputs):
            neuron.output = soft_out

        max_output = max(neuron.output for neuron in self.output_layer)
        activated_neuron_index = next(i for i, neuron in enumerate(self.output_layer) if neuron.output == max_output)

        input_vals = [i.value for i in self.inputs]
        hidden_vals = [n.output for layer in self.hidden_layers for n in layer]
        output_vals = [n.output for n in self.output_layer]
        self.plot_brain(input_vals, hidden_vals, output_vals)

        return self.outputs[activated_neuron_index]
    
    def copy_and_mutate(self, parent_brain):
        # Copy the parent brain
        self.hidden_layers = copy.deepcopy(parent_brain.hidden_layers)
        self.output_layer = copy.deepcopy(parent_brain.output_layer)

        # Mutate weights and biases
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.weights = [w + random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH) for w in neuron.weights]
                neuron.bias += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)

        for neuron in self.output_layer:
            neuron.weights = [w + random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH) for w in neuron.weights]
            neuron.bias += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
    
def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + math.exp(-x))

def relu(x):
    # ReLU activation function
    return max(0, x)