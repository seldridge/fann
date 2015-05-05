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
				break;
			case FANN_CONVOLUTIONAL_LAYER://has a bias neuron in previous layer
				num_connections = ((layers[j].conv_size)*num_neurons_out)+num_neurons_out;
				break;
			case FANN_FCONNECTED_LAYER:
				num_connections = (num_neurons_in + 1) * num_neurons_out;
				break;
		}
		if (layers[j].type == FANN_FCONNECTED_LAYER){
			connections_per_neuron = num_neurons_in;
		}else{
			connections_per_neuron = layers[j].conv_size;
		}
		allocated_connections = 0;
		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++)
		{
			layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
			allocated_connections += connections_per_neuron;
			layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;
			if (layers[j].type != FANN_MAXPOOLING){
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
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			{
				ann->connections[neuron_it->first_con] = bias_neuron;
				ann->connections_to_weights[neuron_it->first_con] = num_weights;
				ann->weights[num_weights] = (fann_type) fann_random_bias_weight();
			}
			offset = 0;
			/* then connect the rest of the unconnected neurons */
			last_neuron = layer_it->last_neuron - 1;
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
					ann->weights[tmp_con] = (fann_type) fann_random_bias_weight();
					ann->connections[tmp_con] = (layer_it - 1)->first_neuron + (tmp_con - neuron_it->first_con);
					num_weights+=i;
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


int main()
{
	const unsigned int num_input = 4;
	const unsigned int num_output = 2;
	const unsigned int num_layers = 4;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0.001;
	const unsigned int max_epochs = 500000;
	const unsigned int epochs_between_reports = 1000;
	struct fann_layer_type *layers = fann_create_layer_types(num_layers);
	fann_add_layer_type(layers, 0, FANN_INPUT_LAYER, num_input, 0, 0);
	fann_add_layer_type(layers, 1, FANN_CONVOLUTIONAL_LAYER, 3, 2, 1);
	fann_add_layer_type(layers, 2, FANN_MAXPOOLING_LAYER, 2, 2, 1);
	fann_add_layer_type(layers, 3, FANN_FCONNECTED_LAYER, num_output, 0, 0);


	struct fann *ann = fann_create_conv(num_layers, layers);

	/*fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);

	fann_save(ann, "xor_float.net");

	fann_destroy(ann);

	return 0;*/
}
