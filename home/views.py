from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
from .pre_proccess_data import pre_proccess_data
from django.contrib import messages
# Tải mô hình đã lưu
model = joblib.load('random_forest_model.joblib')

def home(request):
    return render(request, 'home/home.html')

def predict(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        is_married = int(request.POST.get('is_married'))
        ticket_trequency = int(request.POST.get('ticket_frequency'))
        pclass = int(request.POST.get('pclass'))
        sex = int(request.POST.get('sex'))
        deck = int(request.POST.get('deck'))
        embarked = int(request.POST.get('embarked'))
        title = 1 if sex == 1 else 2
        family_sizes = int(request.POST.get('family_sizes'))
        fare = float(request.POST.get('fare'))


        passenger_dict = {
            'Age': [age], 
            'Is_Married': [is_married], 
            'Ticket_Frequency': [ticket_trequency],
            'Pclass': [pclass], 
            'Sex': [sex], 
            'Deck': [deck], 
            'Embarked': [embarked], 
            'Title': [title], 
            'Family_Size_Grouped': [family_sizes],
            'Fare': [fare],
        }

        data_test = [pre_proccess_data(passenger_dict)]

        
        # Dự đoán nhãn của mẫu mới
        survived_list = model.predict(data_test)
        if survived_list[0] == 1:
            messages.success(request, "Chào mừng bạn đến với chuyến tàu của chúng tôi")
        elif survived_list[0] == 0:
            messages.error(request, "Chúng tôi dự đoán bạn sẽ chết nếu như lên chuyến tàu này")
        

    # Trả về kết quả dự đoán dưới dạng JSON
    return render(request=request, template_name='home/home.html', context={'passenger_dict': passenger_dict} )




