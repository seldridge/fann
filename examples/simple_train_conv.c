/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "fann.h"

struct fann *fann_create_conv(unsigned int num_layers, struct fann_layer_type *layers);
struct fann *fann_create_conv_array(float connection_rate,
															 unsigned int num_layers,
															 struct fann_layer_type *layers);
fann_type *cfann_run(struct fann * ann, fann_type * input);

struct fann *fann_create_conv(unsigned int num_layers, struct fann_layer_type *layers)
{
	struct fann *ann;
	int i;
	int status;
	int arg;

	ann = fann_create_conv_array(1, num_layers, layers);

	free(layers);

	return ann;
}
struct fann *fann_create_conv_array(float connection_rate,
															 unsigned int num_layers,
															 struct fann_layer_type *layers)
{
	struct fann_layer *layer_it, *last_layer, *prev_layer;
	struct fann *ann;
	struct fann_neuron *neuron_it, *last_neuron, *random_neuron, *bias_neuron;
#ifdef DEBUG
	unsigned int prev_layer_size;
#endif
	unsigned int num_neurons_in, num_neurons_out, i, j, k, offset, l, stride, num_weights;
	unsigned int min_connections, max_connections, num_connections;
	unsigned int connections_per_neuron, allocated_connections;
	unsigned int random_number, found_connection, tmp_con;

#ifdef FIXEDFANN
	unsigned int multiplier;
#endif
	printf("Beginning to create conv net\n");
	/* seed random */
#ifndef FANN_NO_SEED
	fann_seed_rand();
#endif
	printf("Allocating structure\n");
	/* allocate the general structure */
	ann = fann_allocate_structure(num_layers);
	printf("Allocatted structure\n");
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	printf("Set connection rate");
#ifdef FIXEDFANN
	multiplier = ann->multiplier;
	fann_update_stepwise(ann);
#endif
	printf("determine how many neurons there should be in each layer");
	/* determine how many neurons there should be in each layer */
	i = 0;
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		/* we do not allocate room here, but we make sure that
		 * last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		if (layer_it != ann->last_layer - 1){//check next layer to see if bias is necessary
			if (layers[i+1].type != FANN_MAXPOOLING_LAYER){
				layer_it->last_neuron = layer_it->first_neuron + layers[i++].size + 1;	/* +1 for bias */
			}else{
				layer_it->last_neuron = layer_it->first_neuron + layers[i++].size;
			}
		}
		else{//assuming we end in fconnected
			layer_it->last_neuron = layer_it->first_neuron + layers[i++].size + 1;	/* +1 for bias */
		}
		ann->total_neurons += (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
	}

	ann->num_output = (unsigned int)((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron - 1);
	ann->num_input = (unsigned int)(ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);
	printf("allocate room for the actual neurons %d", ann->total_neurons);
	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}
	printf("Calculating num connections and weights");
	num_neurons_in = ann->num_input;
	j = 1;
	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
	{
		num_neurons_out = layers[j].size;
		/*�if all neurons in each layer should be connected to at least one neuron
		 * in the previous layer, and one neuron in the next layer.
		 * and the bias node should be connected to the all neurons in the next layer.
		 * Then this is the minimum amount of neurons */
		min_connections = fann_max(num_neurons_in, num_neurons_out); /* not calculating bias */
		max_connections = num_neurons_in * num_neurons_out;	     /* not calculating bias */
		switch (layers[j].type)
		{
			case FANN_MAXPOOLING_LAYER://does NOT have a bia neuron in previous layer
				num_connections = (layers[j].conv_size)*num_neurons_out;
				connections_per_neuron = layers[j].conv_size;
				break;
			case FANN_CONVOLUTIONAL_LAYER://has a bias neuron in previous layer
				num_connections = ((layers[j].conv_size)*num_neurons_out)+num_neurons_out;
				connections_per_neuron = layers[j].conv_size + 1;
				break;
			case FANN_FCONNECTED_LAYER:
				num_connections = (num_neurons_in + 1) * num_neurons_out;
				connections_per_neuron = num_neurons_in + 1;
				break;
		}
		allocated_connections = 0;
		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++)
		{
			layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
			allocated_connections += connections_per_neuron;
			layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;
			if (layers[j].type != FANN_MAXPOOLING_LAYER){
				layer_it->first_neuron[i].activation_function = FANN_SIGMOID_STEPWISE;
			}else{
				layer_it->first_neuron[i].activation_function = FANN_MAXPOOLING;
			}
#ifdef FIXEDFANN
			layer_it->first_neuron[i].activation_steepness = ann->multiplier / 2;
#else
			layer_it->first_neuron[i].activation_steepness = 0.5;
#endif
		}
		if (layer_it != ann->last_layer - 1){
			if (layers[j+1].type != FANN_MAXPOOLING_LAYER){
				/* bias neuron also gets stuff */
				layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
				layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;
			}
		}else{
			layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
			layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;
		}
		ann->total_connections += num_connections;
		switch (layers[j].type)
		{
			case FANN_MAXPOOLING_LAYER://does NOT have a bia neuron in previous layer
				ann->num_weights += layers[j].conv_size + 1;
				break;
			case FANN_CONVOLUTIONAL_LAYER://has a bias neuron in previous layer
				ann->num_weights += layers[j].conv_size;
				break;
			case FANN_FCONNECTED_LAYER:
				ann->num_weights += num_connections;
				break;
		}
		/* used in the next run of the loop */
		num_neurons_in = num_neurons_out;
		j++;
	}
	printf("Allocating connections");
	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}
	/* For convolutional layers*/
	j = 1;
	printf("Connecting layers");
	num_weights = 0;
	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
	{
		num_neurons_out = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron - 1);
		num_neurons_in = (unsigned int)((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron - 1);
		if (layers[j].type == FANN_CONVOLUTIONAL_LAYER){
			/* first connect the bias neuron */
			printf("Connecting conv");
			bias_neuron = (layer_it - 1)->last_neuron - 1;
			last_neuron = layer_it->last_neuron - 1;
			last_neuron = layer_it->last_neuron - 1;
			if (layer_it != ann->last_layer - 1){
				if (layers[j+1].type == FANN_MAXPOOLING_LAYER){
					/* bias neuron also gets stuff */
					last_neuron = layer_it->last_neuron;
				}
			}
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			{
				ann->connections[neuron_it->first_con] = bias_neuron;
				ann->connections_to_weights[neuron_it->first_con] = num_weights;
				ann->weights[num_weights] = (fann_type) fann_random_bias_weight();
			}
			offset = 0;
			/* then connect the rest of the unconnected neurons */
			last_neuron = layer_it->last_neuron - 1;
			if (layer_it != ann->last_layer - 1){
				if (layers[j+1].type == FANN_MAXPOOLING_LAYER){
					/* bias neuron also gets stuff */
					last_neuron = layer_it->last_neuron;
				}
			}
			stride = layers[j].stride;
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			{
				l = offset;
				/* find empty space in the connection array and connect */
				for(i = 1; i != neuron_it->last_con - neuron_it->first_con; i++)
				{
					/* we have found a neuron that is not allready
					 * connected to us, connect it */
					ann->connections[neuron_it->first_con + i] = (layer_it - 1)->first_neuron + l;
					ann->connections_to_weights[neuron_it->first_con + i] = num_weights + i;
					ann->weights[num_weights + i] = (fann_type) fann_random_weight();
					l++;
				}
				offset+=stride;
			}
			num_weights+=layers[j].conv_size + 1;
		}
		if (layers[j].type == FANN_MAXPOOLING_LAYER){
			printf("Connecting maxpool");
			offset = 0;
			//theres no bias neuron to connect here
			last_neuron = layer_it->last_neuron - 1;
			if (layer_it != ann->last_layer - 1){
				if (layers[j+1].type != FANN_MAXPOOLING_LAYER){
					/* bias neuron also gets stuff */
					last_neuron = layer_it->last_neuron;
				}
			}
			stride = layers[j].stride;
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			{
				l = offset;
				/* find empty space in the connection array and connect */
				for(i = 0; i != neuron_it->last_con - neuron_it->first_con; i++)
				{
					/* we have found a neuron that is not allready
					* connected to us, connect it */
					ann->connections[neuron_it->first_con + i] = (layer_it - 1)->first_neuron + l;
					ann->connections_to_weights[neuron_it->first_con + i] = num_weights + i;
					ann->weights[num_weights + i] = (fann_type) 1;
					l++;
				}
				offset+=stride;
			}
			num_weights+=layers[j].conv_size;
		}
		if(layers[j].type == FANN_FCONNECTED_LAYER)
		{
			printf("Connecting fconnect");
				last_neuron = layer_it->last_neuron - 1;
				if (layer_it != ann->last_layer - 1){
					if (layers[j+1].type != FANN_MAXPOOLING_LAYER){
						/* bias neuron also gets stuff */
						last_neuron = layer_it->last_neuron;
					}
				}
				for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
				{
					tmp_con = neuron_it->last_con - 1;
					k = 0;
					for(i = neuron_it->first_con; i != tmp_con; i++)
					{
						ann->weights[num_weights+k] = (fann_type) fann_random_weight();
						/* these connections are still initialized for fully connected networks, to allow
						 * operations to work, that are not optimized for fully connected networks.
						 */
						ann->connections_to_weights[i] = num_weights+k;
						ann->connections[i] = (layer_it - 1)->first_neuron + (i - neuron_it->first_con);
						k++;
					}

					/* bias weight */
					ann->weights[num_weights+k] = (fann_type) fann_random_bias_weight();
					ann->connections_to_weights[tmp_con] = num_weights + k;
					ann->connections[tmp_con] = (layer_it - 1)->first_neuron + (tmp_con - neuron_it->first_con);
					num_weights+=k;
				}
		}
		j++;

#ifdef DEBUG
		printf("  layer       : %d neurons, 1 bias\n", num_neurons_out);
#endif
	}

		/* TODO it would be nice to have the randomly created
		 * connections sorted for smoother memory access.
		 */

#ifdef DEBUG
	printf("output\n");
#endif

	return ann;
}

fann_type *cfann_run(struct fann * ann, fann_type * input)
{
	struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
	unsigned int i, num_connections, num_input, num_output, counter;
	fann_type neuron_sum, *output;
	fann_type *weights;
	struct fann_layer *layer_it, *last_layer;
	unsigned int activation_function;
	fann_type steepness;
	unsigned int *connections_to_weights;

	/* store some variabels local for fast access */
	struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

#ifdef FIXEDFANN
	int multiplier = ann->multiplier;
	unsigned int decimal_point = ann->decimal_point;

	/* values used for the stepwise linear sigmoid function */
	fann_type r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0;
	fann_type v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0;

	fann_type last_steepness = 0;
	unsigned int last_activation_function = 0;
#else
	fann_type max_sum = 0;
#endif
	counter = 0;
	/* first set the input */
	num_input = ann->num_input;
	for(i = 0; i != num_input; i++)
	{
#ifdef FIXEDFANN
		if(fann_abs(input[i]) > multiplier)
		{
			printf
				("Warning input number %d is out of range -%d - %d with value %d, integer overflow may occur.\n",
				 i, multiplier, multiplier, input[i]);
		}
#endif
		first_neuron[i].value = input[i];
	}
	/* Set the bias neuron in the input layer */
#ifdef FIXEDFANN
	(ann->first_layer->last_neuron - 1)->value = multiplier;
#else
	(ann->first_layer->last_neuron - 1)->value = 1;
#endif

	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{

			if(neuron_it->first_con == neuron_it->last_con)
			{
				/* bias neurons */
#ifdef FIXEDFANN
				neuron_it->value = multiplier;
#else
				neuron_it->value = 1;
#endif
				continue;
			}

			activation_function = neuron_it->activation_function;
			steepness = neuron_it->activation_steepness;

			neuron_sum = 0;
			num_connections = neuron_it->last_con - neuron_it->first_con;
			weights = ann->weights;
			connections_to_weights = ann->connections_to_weights + neuron_it->first_con;

			if(ann->connection_rate >= 1)
			{
				if(ann->network_type == FANN_NETTYPE_SHORTCUT)
				{
					neurons = ann->first_layer->first_neuron;
				}
				else
				{
					neurons = (layer_it - 1)->first_neuron;
				}


				/* unrolled loop start */
				i = num_connections & 3;	/* same as modulo 4 */
				switch (i)
				{
					case 3:
						neuron_sum += fann_mult(weights[connections_to_weights[2]], neurons[2].value);
					case 2:
						neuron_sum += fann_mult(weights[connections_to_weights[1]], neurons[1].value);
					case 1:
						neuron_sum += fann_mult(weights[connections_to_weights[0]], neurons[0].value);
					case 0:
						break;
				}
				counter++;
				for(; i != num_connections; i += 4)
				{
					neuron_sum +=
						fann_mult(weights[connections_to_weights[i]], neurons[i].value) +
						fann_mult(weights[connections_to_weights[i + 1]], neurons[i + 1].value) +
						fann_mult(weights[connections_to_weights[i + 2]], neurons[i + 2].value) +
						fann_mult(weights[connections_to_weights[i + 3]], neurons[i + 3].value);
				}
				/* unrolled loop end */

				/*
				 * for(i = 0;i != num_connections; i++){
				 * printf("%f += %f*%f, ", neuron_sum, weights[i], neurons[i].value);
				 * neuron_sum += fann_mult(weights[i], neurons[i].value);
				 * }
				 */
			}
			else
			{
				if (activation_function != FANN_MAXPOOLING){
					neuron_pointers = ann->connections + neuron_it->first_con;
					counter++;
					i = num_connections & 3;	/* same as modulo 4 */
					switch (i)
					{
						case 3:
							neuron_sum += fann_mult(weights[connections_to_weights[2]], neuron_pointers[2]->value);
						case 2:
							neuron_sum += fann_mult(weights[connections_to_weights[1]], neuron_pointers[1]->value);
						case 1:
							neuron_sum += fann_mult(weights[connections_to_weights[0]], neuron_pointers[0]->value);
						case 0:
							break;
					}

					for(; i != num_connections; i += 4)
					{
						neuron_sum +=
							fann_mult(weights[connections_to_weights[i]], neuron_pointers[i]->value) +
							fann_mult(weights[connections_to_weights[i + 1]], neuron_pointers[i + 1]->value) +
							fann_mult(weights[connections_to_weights[i + 2]], neuron_pointers[i + 2]->value) +
							fann_mult(weights[connections_to_weights[i + 3]], neuron_pointers[i + 3]->value);
					}
				}else{
					for(i = 0; i != num_connections; i++)
					{
						neuron_sum = fann_max(neuron_pointers[i]->value, neuron_sum);
					}
				}
			}

#ifdef FIXEDFANN
			neuron_it->sum = fann_mult(steepness, neuron_sum);

			if(activation_function != last_activation_function || steepness != last_steepness)
			{
				switch (activation_function)
				{
					case FANN_SIGMOID:
					case FANN_SIGMOID_STEPWISE:
						r1 = ann->sigmoid_results[0];
						r2 = ann->sigmoid_results[1];
						r3 = ann->sigmoid_results[2];
						r4 = ann->sigmoid_results[3];
						r5 = ann->sigmoid_results[4];
						r6 = ann->sigmoid_results[5];
						v1 = ann->sigmoid_values[0] / steepness;
						v2 = ann->sigmoid_values[1] / steepness;
						v3 = ann->sigmoid_values[2] / steepness;
						v4 = ann->sigmoid_values[3] / steepness;
						v5 = ann->sigmoid_values[4] / steepness;
						v6 = ann->sigmoid_values[5] / steepness;
						break;
					case FANN_SIGMOID_SYMMETRIC:
					case FANN_SIGMOID_SYMMETRIC_STEPWISE:
						r1 = ann->sigmoid_symmetric_results[0];
						r2 = ann->sigmoid_symmetric_results[1];
						r3 = ann->sigmoid_symmetric_results[2];
						r4 = ann->sigmoid_symmetric_results[3];
						r5 = ann->sigmoid_symmetric_results[4];
						r6 = ann->sigmoid_symmetric_results[5];
						v1 = ann->sigmoid_symmetric_values[0] / steepness;
						v2 = ann->sigmoid_symmetric_values[1] / steepness;
						v3 = ann->sigmoid_symmetric_values[2] / steepness;
						v4 = ann->sigmoid_symmetric_values[3] / steepness;
						v5 = ann->sigmoid_symmetric_values[4] / steepness;
						v6 = ann->sigmoid_symmetric_values[5] / steepness;
						break;
					case FANN_THRESHOLD:
						break;
				}
			}

			switch (activation_function)
			{
				case FANN_SIGMOID:
				case FANN_SIGMOID_STEPWISE:
					neuron_it->value =
						(fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, 0,
												  multiplier, neuron_sum);
					break;
				case FANN_SIGMOID_SYMMETRIC:
				case FANN_SIGMOID_SYMMETRIC_STEPWISE:
					neuron_it->value =
						(fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6,
												  -multiplier, multiplier, neuron_sum);
					break;
				case FANN_THRESHOLD:
					neuron_it->value = (fann_type) ((neuron_sum < 0) ? 0 : multiplier);
					break;
				case FANN_THRESHOLD_SYMMETRIC:
					neuron_it->value = (fann_type) ((neuron_sum < 0) ? -multiplier : multiplier);
					break;
				case FANN_LINEAR:
					neuron_it->value = neuron_sum;
					break;
				case FANN_LINEAR_PIECE:
					neuron_it->value = (fann_type)((neuron_sum < 0) ? 0 : (neuron_sum > multiplier) ? multiplier : neuron_sum);
					break;
				case FANN_LINEAR_PIECE_SYMMETRIC:
					neuron_it->value = (fann_type)((neuron_sum < -multiplier) ? -multiplier : (neuron_sum > multiplier) ? multiplier : neuron_sum);
					break;
				case FANN_ELLIOT:
				case FANN_ELLIOT_SYMMETRIC:
				case FANN_GAUSSIAN:
				case FANN_GAUSSIAN_SYMMETRIC:
				case FANN_GAUSSIAN_STEPWISE:
				case FANN_SIN_SYMMETRIC:
				case FANN_COS_SYMMETRIC:
					fann_error((struct fann_error *) ann, FANN_E_CANT_USE_ACTIVATION);
					break;
				case FANN_MAXPOOLING:
					neuron_it->value = neuron_sum;
					break;
			}
			last_steepness = steepness;
			last_activation_function = activation_function;
#else
			if (activation_function != FANN_MAXPOOLING){
				neuron_sum = fann_mult(steepness, neuron_sum);

				max_sum = 150/steepness;
				if(neuron_sum > max_sum)
					neuron_sum = max_sum;
				else if(neuron_sum < -max_sum)
					neuron_sum = -max_sum;

				neuron_it->sum = neuron_sum;
			}
			fann_activation_switch(activation_function, neuron_sum, neuron_it->value);
#endif
		}
	}

	/* set the output */
	output = ann->output;
	num_output = ann->num_output;
	neurons = (ann->last_layer - 1)->first_neuron;
	for(i = 0; i != num_output; i++)
	{
		output[i] = neurons[i].value;
	}
	return ann->output;
}


int main()
{
	const unsigned int num_input = 4;
	const unsigned int num_output = 2;
	const unsigned int num_layers = 4;
	const float desired_error = (const float) 0.001;
	const unsigned int max_epochs = 5000000;
	const unsigned int epochs_between_reports = 1000;
	unsigned int i = 0;
	struct fann_layer_type *layers = fann_create_layer_types(num_layers);
	fann_add_layer_type(layers, 0, FANN_INPUT_LAYER, num_input, 0, 0);
	fann_add_layer_type(layers, 1, FANN_CONVOLUTIONAL_LAYER, 3, 2, 1);
	fann_add_layer_type(layers, 2, FANN_MAXPOOLING_LAYER, 2, 2, 1);
	fann_add_layer_type(layers, 3, FANN_FCONNECTED_LAYER, num_output, 0, 0);

	for (i = 0; i < 10; i++){
		struct fann *ann = fann_create_conv(num_layers, layers);

		fann_destroy(ann);
	}


	/*fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);*/




	return 0;
}
