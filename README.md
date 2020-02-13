# Automated-Voting
Automatic design of voting rules using neural networks and the genetic algorithm. 

## Next steps

- [ ] Compile model, training and loss into a class
- [ ] Write simulator generator for IM matrices
- [ ] Allow for multiple forward passes to happen in the loss function for simulators
- [ ] Add IM component to loss

## TODO 

1. Voting
    - [x] Profile generator
    - [x] Universal input preprocessing (make sure matrix is properly transposed)
    - [x] Separate constraint/checks (i.e is there a condorcet winner?)
    - [ ] Alternative scenarios generator - Individual Manipulation (IM)
    - [ ] Alternative scenarios generator - Coalition Manipulation (CM)
    - [ ] Alternative scenarios generator - IIA

2. Neural network
    - [x] Finish basic Keras tutorial
    - [x] Write basic architecture for voting NN
    - [x] Write training procedure 
    - [x] Write input component of pipeline - make it ready for any transformation 
    - [x] Loss component - existing candidates (Condorcet, majority, plurality)
    - [ ] Loss component - IM
    - [ ] Loss component - CM
    - [ ] Loss component - IIA 
    - [x] Full Keras implementation/tensorflow 2.0 
    - [ ] Full Pytorch implementation

3. Evaluators/Variables to keep track of
    - [ ] IM: ratio of predicted candidates to true winner
    - [ ] IM: number of simulations per profile vs performance 
    - [ ] IIA: ratio of 

4. Genetic Algorithm
    - [x] Basic implementation 
    - [ ] Add all constraint functions
    - [ ] Make it more fancy/run experiments