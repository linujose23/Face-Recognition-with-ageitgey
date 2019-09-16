import face_recognition

from PIL import Image,ImageDraw

#reading image
image_of_bill = face_recognition.load_image_file('Bill Gate.jpg')
#getting the face encodings 
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]


image_of_steve = face_recognition.load_image_file('Steve Jobs.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]


image_of_elon_musk = face_recognition.load_image_file('Elon Musk.jpg')
elonmusk_face_encoding = face_recognition.face_encodings(image_of_elon_musk)[0]


known_face_encodings = [

  bill_face_encoding,
  steve_face_encoding,
  elonmusk_face_encoding
  
]

known_face_names = [

  "Bill Gates",
  "Steve Jobs",
  "Elon Musk"

]


test_image = face_recognition.load_image_file('bill-steve-elon.jpg')


face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image,face_locations)


pil_image = Image.fromarray(test_image)


draw = ImageDraw.Draw(pil_image)


for(top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):

    matches = face_recognition.compare_faces(known_face_encodings,face_encoding)

    name = "unknown persons"


    if True in matches:

        first_match_index = matches.index(True)

        name = known_face_names[first_match_index]


        # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

    # Draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('identified.jpg')