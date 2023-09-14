# InfoGatheringPOMDPs
Repo for defining, solving and analyzing information-gathering pomdps. We define an information gathering POMDP as one where the transition dynamics are trivial and the actions are primarily to gather information to better infer the state in order to make a final decisions which will be economic or non-economic. Examples of information gathering POMDPs are [mineral exploration](https://gmd.copernicus.org/articles/16/289/2023/) and geothermal reservoir assessment (example provided in code). 

## Installation
Clone the repo
```
git clone https://github.com/ancorso/InfoGatheringPOMDPs.git
```

Add (or `dev`) the package to julia. Start `julia`, then
```
]add YOUR/FULL/PATH/TO/InfoGatheringPOMDPs
```

The primary solver used is SARSOP. Install [this fork](https://github.com/ancorso/NativeSARSOP.jl) of the NativeSARSOP.jl package as it implements an action-dependent discounting which is crucial for good performance of the SARSOP algorithm

## Running an example
To run the example geothermal case, navigate to the `InfoGatheringPOMDPs` folder and call
```
julia examples/geothermal_example.jl
```

## Future Work
* Make a version of the SARSOP policy that targets a particular PES


Maintained by Anthony Corso (acorso@stanford.edu)