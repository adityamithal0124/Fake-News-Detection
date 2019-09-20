import pickle 

def fake_news():
    statement = input("Please enter the statement: ")
    model = pickle.load(open('final_model.sav','rb'))
    prediction = model.predict([statement])
    probability = model.predict_proba([statement])
    
    return (print("Prediction: ",prediction[0]), print("Probability: ",probability[0][1]))

fake_news()

