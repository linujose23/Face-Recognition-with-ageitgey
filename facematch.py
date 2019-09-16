import face_recognition


image_of_bill = face_recognition.load_image_file('Bill Gate.jpg')

bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]


unknown_image = face_recognition.load_image_file('bill-gates-4.jpg')

unknown_face_encodings = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([bill_face_encoding],unknown_face_encodings)


if results[0]:

    print('this is bill')

else :

    print('this isnt bill!')

