    Adversarial manipulation of machine learning models is a growing threat to the security of machine learning models. This is due to the fact that machine learning models are becoming more and more complex and are being used in more and more critical applications. This means that the consequences of an attack on a machine learning model can be more severe than an attack on a traditional software application. This is because machine learning models are often used to make decisions that have a direct impact on the real world, such as whether or not a person is approved for a loan, etc.
    
    
    In  data access attacks, initial dataset used to create the legitimate model can be used to create a substitute model. This substitute model can be used to test different attack types under testing. These attacks are usually convayed by malicious actors that sell data to other malicious actors. Data leaked from cyber attacks on telecommunication companies can be exploited for these attacks.

    Data access attacks can be defented from different authentication and authorisation mechanisms. Furthermore, standard at rest encryption such as AES256 and at transit encryption such as TLS can be used to protect data from being accessed by unauthorised users.


    Indirect poisoning refers to type of attack that the malicious actors gain access to raw data and change the nature or data to give incorrect inference results. This will be done before or after data collection and before preprocessing. This could have serious consequences for the companies, such as financial losses or damage to its reputation. Typical ways that malicious actors carry indirect poisoning is via gaining data access.

    To protect against indirect poisoning attacks, data must be validated before preprocessing. These attacks are harder to defend against as the malicious actors can change the data before they are collected. It is important that the dats used are from trusted and verified sources.

    Direct poisoning attacks are carried out by gaining access to the preprocessed data and altering models, data or results. There are few ways direct poisoning can be carried out; data injection, data manipulation and logic manipulation. Data injections and data manipulation is done by changing the training or testing data. Logic manipulation is done by getting access to computing systems or code repository and changing the code.

    Logic manipulation can be mitigated via builing layered software architectures, using proper at rest and in transit encryption. Data injection and manipulation can be mitigated by using proper data validation and sanitization. Extra care must be taken when using public data or input sources to prevent data injection and data manipulation.


    Evasion attacks focus on tampering with the input data to model to produce incorrect inference results. Algorithms for evasion attacks require knowledge about the model or substitute models. Evasion attacks can be difficult to detect and defend against, as they often involve subtle changes to inputs that are not immediately noticeable.

    To protect against evasion attacks, it is important to implement robust security controls and to regularly test and update them to ensure that they are effective at detecting and blocking these types of attacks. This may involve implementing input validation and filtering, as well as using security tools such as firewalls and intrusion detection and prevention systems.


    Oracle attacks are carried out using an API to input data and observe the output of the model. This can be done by using a substitute model to test the model. This attack is similar to evasion attacks, but the goal is to find the input that produces the desired output. This attack is usually carried out by malicious actors that have access to the model. This does not require direct knowledge about the model. Attacker will associate input with outputs to identify input combination that model will create incorrect output. 

    Similar to evasion attacks, oracle attacks are also difficult to mitigate. However using robust models trained with noise can help mitigate this attack. Masking the inputs and gradient can also help mitigate this attack.
