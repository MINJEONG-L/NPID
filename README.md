# NPID  

## Background  
### class  
  - An object or group of objects with similar properties  
  - In image classification, the category of the target want to predict  
   ex) cars, dogs, cats, palnes, etc  
### instance
  - Specific examples of classes or individual objects of classes
  - In image classfication, a representation of an object or particular object in an image  

## NPID: Unsupervised Feature Learning via Non-Parametric Instance Discrimination [Z. Wu et al., 2018]
### Goal  
  - Capturing the features of data well with unsupervised learning and achieving superior performance in supervised learning tasks
### Motivations  
  - Supervised learning results that discover apparent similarity among semantic categories without being explicitly guided to do so
      - Learning how to distinguish differences between individual instances independently of sementic categories
  - Using unsupervised learning effectively as instance-level discrimination
      - Learning how to discriminate differences of similarities between input data
### Contributions  
  - Solving the problem of the parametric softmax using the non-parametric approach
      - Generalizing predictable for new classes as well
  - Solving high costs of softmax with Noise-Contrastive Estimation and memory bank
      - Changing multi-class classification problems into binary classification problems
      - Performing relatively simply
### Methods  
#### Approach  
  - Learning an embedding function $v=𝒇_𝜽(𝒙)$ without supervision  
  - Novel unsupervised feature learning approach wwith Instance-level discrimination
      - Treating each image instance as a distinct class of its own
      - Training a classifier to distinguish between individual instance classes
  - $𝒅_𝜽 (𝒙,𝒚)=‖𝒇_𝜽 (𝒙)−𝒇_𝜽 (𝒚)‖$
      - $𝒇_𝜽$: a deep neural network with parameters $𝜽$  
      - $𝒙$: mapping image  
      - $𝒗$: feature vector
    ![image](https://github.com/MINJEONG-L/NPID/assets/82145878/6d62fea2-0af2-4846-b20a-e2cbd1080376)

#### Non-parametric softmax classifier  
  - Parametric classifier
      - Formulating the instance-level classification objective using the softmax criterion
      - $𝑷(𝒊│𝒗)=(𝒆𝒙𝒑(𝒘_𝒊^𝑻 𝒗))/(∑2_(𝒋=𝟏)^𝒏▒〖𝒆𝒙𝒑(𝒘_𝒋^𝑻 𝒗)〗)$
          - $𝒗_𝒊$: feature of the $𝒊$-th image  
          - $𝒘_𝒋^𝑻$: transpose of the weight vector for class $𝒋$  
          - Measuring the probability that the feature vector 𝒗 belongs to the $𝒊$-th instance  
  - Non-parametric classifier
      - Resolving the issues with the parametric softmax formulation
          - Hiding the explicit comparison between instances through 𝒘 acting as a class prototype
          - $𝑷(𝒊│𝒗)=(𝒆𝒙𝒑((𝒗_𝒊^𝑻 𝒗)/𝝉))/(∑2_(𝒋=𝟏)^𝒏▒〖𝒆𝒙𝒑((𝒗_𝒋^𝑻 𝒗)/𝝉)〗)!$
            - $𝝉$: temperature parameter
            - Replacing $𝒘_𝒋^𝑻 𝒗$ with $𝒗_𝒋^𝑻 𝒗$
          - $𝑱(𝜽)=−∑_(𝒊=𝟏)^𝒏▒〖𝐥𝐨𝐠𝑷(𝒊|𝒇_𝜽 (𝒙_𝒊 ))〗$
          - Non-parametric softmax formulation facilitates comparison between instances
    - Memory bank
        - Requiring to cumpute the probability $𝑷(𝒊│𝒗), {𝒗_𝒋}$ for all images
        - Memory Bank $𝑽={𝒗_𝒋}$
            - Initializing unit random vector and store it in $𝑽$
            - Updating with $𝜽$ for each learning iteration
    - Discussions
        - Better generalization performance
            - Learning feature presentation and the corresponding metric, not a specific class
            - Eliminating the need for computing and storing the gradients for {$𝒘_𝒋$}
                - Making it more scalable for big data aplications
      
#### Noise-contrastive estimation  





