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
  - Prohibitive cost with non-parametric softmax in the situation where n is very large
      - Using Noise-contrastive estimation to approximate the full softmax
  - Casting the multi-class classification into the binary classification
      - Discriminating between data samples and noise samples
  - Formalizing the noise distribution as a uniform distribution: $𝑷_𝒏=𝟏/𝒏$
  - Assuming that noise samples are $𝒎$ times more frequent than data samples
  - $𝒉(𝒊,𝒗)≔𝑷(𝑫=𝟏│𝒊,𝒗)=(𝑷(𝒊│𝒗))/(𝑷(𝒊│𝒗)+𝒎𝑷_𝒏 (𝒊) $
      - Posterior probability of sample 𝒊 with feature 𝒗 being from the data distribution  
      - Correction of the probability that a 𝒊 and 𝒗 are from the data distribution to the probability that they are from the noise distribution  
      - Training objective is to minimize the negative log-posterior distribution of data and noise sample.  
      - $𝑱_𝑵𝑪𝑬 (𝜽)=−𝑬_(𝑷_𝒅 ) [𝐥𝐨𝐠𝒉(𝒊, 𝒗)]−𝒎·〖𝑬_𝑷〗_𝒏 [𝐥𝐨𝐠⁡(𝟏 − 𝒉(𝒊, 𝒗 ′))]$
          - $𝑷_𝒅$: actual data distribution  
          - $𝑷_𝒏$: noise distribution 
          - $𝒗′$: feature from randomly sampled according to $𝑷_𝒏$

#### Proximal Regularization  
  - Only having one instance per class unlike typical classification settings where each class has many instances
      - Oscillating learning process with lack of learning or long time to convergence
  - Solving by the proximal regularization method
      - Introducing an additional term to encourage the smoothness of the training dynamics
    ![image](https://github.com/MINJEONG-L/NPID/assets/82145878/e7f341c2-215b-4339-8e49-eca8bf073944)

  - $𝑱_𝑵𝑪𝑬 (𝜽)=−𝑬_(𝑷_𝒅 ) [𝐥𝐨𝐠𝒉(𝒊,𝒗_𝒊^((𝒕−𝟏) ) )−𝝀‖𝒗_𝒊^((𝒕) )−𝒗_𝒊^((𝒕−𝟏) ) ‖_𝟐^𝟐 ]−𝒎 ⋅〖𝑬_𝑷〗_𝒏 [𝐥𝐨𝐠(𝟏−𝒉(𝒊,𝒗^′(𝒕−𝟏)  ))]$
      - 𝒕: current iteration  
      - 𝝀: proximal regularization impact degree
   
#### Weighted k-Nearest Neighbor Classifier 
  - Using Weighted K-Nearest Neighbor Classifier as model for differentiating test images
  - Process for classifying test image 𝒙 ̂
      1. Computing 𝒇 ̂=𝒇_𝜽 (𝒙 ̂) and comparing it against the embeddings in the memory bank using the cosine similarity 𝒔_𝒊  
        - 𝒔_𝒊=𝒄𝒐𝒔⁡(𝒗_𝒊,𝒇 ̂)  
      2. Selecting the top 𝒌 nearest neighbors denoted by 𝑵_𝒌  
        - Making the prediction via weighted voting  
      3. Computing the contributing weight of neighbor 𝒙_𝒊  
        - 𝜶_𝒊=𝒆𝒙𝒑⁡(𝒔_𝒊/𝝉)  
      4. Getting a total weight 𝒘_𝒄, classifying as the highest value class  
        - 𝒘_𝒄=∑_𝒋▒𝜶_𝒊 ⋅𝟏(𝒄_𝒊=𝒄)
### Experiments  
#### Parametric vs. Non-parametric Softmax  
  - Parametric softmax
      - Obtaining accuracy of 60.3% and 63.0% with linear SVM* and kNN* classifiers respectively  
  - Non-parametric softmax
      - Rising accuracy to 75.4% and 80.8% for the linear and nearest neighbor classifiers  
  - NCE* approximating non-parametric softmax
       - Controlling the approximation using m  
       - When m = 4,096, the accuracy approaches that at m = 49,999 – full form evaluation without any approximation
           - Providing assurance that NCE is an efficient approximation.  
    ![image](https://github.com/MINJEONG-L/NPID/assets/82145878/88be13c2-ff1f-446b-a9a6-728d38a39d50)  
    
