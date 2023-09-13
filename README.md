# InfoGatheringPOMDPs
Repo for defining, solving and analyzing Information-gathering pomdps

## Installation
Clone the repo
```
git clone https://github.com/ancorso/InfoGatheringPOMDPs.git
```

Add (or `dev`) the package to julia. Start `julia`, then
```
]add YOUR/FULL/PATH/TO/InfoGatheringPOMDPs
```

## Running an example
To run the example geothermal case, navigate to the `InfoGatheringPOMDPs` folder and call
```
julia examples/geothermal_example.jl
```

## Future Work
* Make a version of the SARSOP policy that targets a particular PES
* Fork NativeSarsop and add proper discounting for the actions
* Flow diagram for actions and abandon


Maintained by Anthony Corso (acorso@stanford.edu)