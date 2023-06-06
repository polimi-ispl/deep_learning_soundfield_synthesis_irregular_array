# Synthesis of Soundfields through Irregular Loudspeaker Arrays Based on Convolutional Neural Networks

# Introduction
Most soundfield synthesis approaches deal with extensive and regular loudspeaker arrays, which are often not suitable for home audio systems, due to  physical space constraints.
In this article we propose a technique for soundfield synthesis through more easily deployable irregular loudspeaker arrays, i.e. where the spacing between loudspeakers is not constant, based on deep learning. The input are the driving signals obtained through a plane wave decomposition-based technique. While the considered driving signals are able to correctly reproduce the soundfield with a regular array, they show degraded performances when using irregular setups. Through a complex-valued Convolutional Neural Network (CNN) we modify the driving signals in order to compensate the errors in the reproduction of the desired soundfield. Since no ground-truth driving signals are available for the compensated ones, we train the model by calculating the loss between the desired soundfield at a number of control points and the one obtained through the driving signals estimated by the network. Numerical results show better reproduction accuracy with respect to the plane wave decomposition-based technique, pressure-matching approach and to linear optimizers for driving signal compensation.

# Method
![real soundfield](/plots/circular_pages/method_train.png)

# Results
You can find the setup as well as additional soundfield plots, other than the ones contained in the paper, at the following pages:

[Irregular Circular Array](/docs/circular.md) 

[Irregular Circular Array (Real Measurements)](/docs/real.md) 

[Irregular Linear Array](/docs/linear.md)

### Support or Contact
For any information regarding the paper or the code send us an email at <luca.comanducci@polimi.it>!

