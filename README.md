# Automated-Voting
Automatic design of voting rules using neural networks and the genetic algorithm. 

## Next steps

- [ ] Look for papers in actual manipulation in voting - or articles?
- [ ] Generate scenarios that allow for manipulation to happen, make existing rules to be manipulable for a good fraction of the time
- [ ] Increase number of possible scenarios
- [ ] Calibrate loss function afterwards
- [ ] Find ways of visualizing voting rules - 2D heatmaps? look for allocation rule plots.
- [ ] Tradeoffs: Less voters, more candidates - what leads to manipulation?
- [ ] If possible look into CM
- [ ] Extra: Is it possible to vectorize stuff?

## Thoughts to consider
- [ ] Why did I or did I not do X in my project? Find the analog of the source of my ideas in my project. Strategy proofness, voting theory, feasibility constraints

## Current experiments

- Learning rate (smaller)
- Regularization 
    - Dividing IM by # of simulations
    - Multiplying individual non-IM components of loss function by # of simulations?

## TODO 

0. General
    - [x] Evaluation metric
   
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
    - [ ] Compare different start names 
    - [ ] Comparel earning over time
    - [ ] Compare against tournament/round based solutions (i.e IRV)

4. Genetic Algorithm
    - [x] Basic implementation 
    - [ ] Add all constraint functions
    - [ ] Make it more fancy/run experiments