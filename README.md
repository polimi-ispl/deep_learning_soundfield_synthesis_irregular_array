# Compensation of Driving Signals for Soundfield Synthesis through Irregular Loudspeaker Arrays Based on Convolutional Neural Networks

# Introduction
We propose a technique for soundfield synthesis using irregular loudspeaker arrays, i.e. where the spacing between loudspeakers is not constant, through a deep learning-based approach. In order to do this, we consider the driving signals obtained through a pre-existing method based on the plane wave decomposition. While the considered driving signal are able to correctly reproduce the soundfield when dealing with a regular array, they show degraded performances when using irregular setups. Through a Convolutional Neural Network (CNN) we modify these driving signals in order to compensate the errors in the reproduction of the desired soundfield. Since no ground-truth driving signals are available for the compensated ones, we train the model by calculating the loss between the desired soundfield at a number of control points and the one obtained through the driving signals estimated by the network. Numerical simulation results show better performances both with respect to the plane wave decomposition-based technique and the pressure-matching approach.

# Method
![real soundfield](/plots/circular_pages/method_train.png)

# Results
You can find the setup as well as additional soundfield plots, other than the ones contained in the paper, at the following pages:

[Irregular Circular Array](/docs/circular.md) 

[Irregular Linear Array](/docs/linear.md)

### Support or Contact
For any information regarding the paper or the code send us an email at <luca.comanducci@polimi.it>!

