# Automated-Voting
Automatic design of voting rules using neural networks and the genetic algorithm. 

## Next steps

- [x] Come up with evaluation metric
    - Possible ideas: for every constraint, consistency of predictions
    - Associate a unit of utility to each preference? Depending on the rank, and see how much utility is being maximized?
- [x] Write code to run existing voting schemes on existing profiles - create a baseline
- [x] Write code for storing generated datasets
- [ ] Extra: See if we can improve manipulation generated profiles 
- [x] Run code to generate datasets using as many possible different distributions 
- [ ] Extend network complexity and write down parameters to improve
- [ ] Run network and existing methods on pregenerated data 


## TODO 

0. General
    - [ ] Evaluation metric
   
1. Voting
    - [x] Profile generator
    - [x] Universal input preprocessing (make sure matrix is properly transposed)
    - [x] Separate constraint/checks (i.e is there a condorcet winner?)
    - [x] Alternative scenarios generator - Individual Manipulation (IM)
    - [ ] Alternative scenarios generator - Coalition Manipulation (CM)
    - [ ] Alternative scenarios generator - IIA

2. Neural network
    - [x] Finish basic Keras tutorial
    - [x] Write basic architecture for voting NN
    - [x] Write training procedure 
    - [x] Write input component of pipeline - make it ready for any transformation 
    - [x] Loss component - existing candidates (Condorcet, majority, plurality)
    - [x] Loss component - IM
    - [ ] Loss component - CM
    - [ ] Loss component - IIA 
    - [x] Full Keras implementation/tensorflow 2.0 
    - [ ] Full Pytorch implementation

3. Evaluators/Variables to keep track of
    - [x] Condorcet/Majority/Plurality winners vs true winners
    - [x] IM: ratio of predicted candidates to true winner
    - [ ] IM: number of simulations per profile vs performance 
    - [ ] IIA: number of times winner changed / number of changes
    - [ ] Different optimizers (SGD, Adam, Adagrad)
    - [ ] Different architectures (Choose 3?)
    - [ ] Compare against tournament/round based solutions (i.e IRV)

4. Genetic Algorithm
    - [x] Basic implementation 
    - [ ] Add all constraint functions
    - [ ] Make it more fancy/run experiments