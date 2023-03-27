# ID-Validator

The data folder holds directories of people's information. Each directory has an ID, a headshot, and an info.txt file. The ID is a picture of the person's ID, the headshot is a picture of the person's headshot, and the info.txt file contains the person's name, date of birth, and other information.

Thor's ID
![Thor's ID](./data/2/id.jpg)

Thor's headshot
![Thor's headshot](./data/2/headshot.jpg)

Thor's info (data/2/info.txt)
```
Thor Hammer Odinson
1997/10/10
```

Once this directory is populated you can run
```
python main.py
```

The output for the example dataset looks like this:
```
Valid: ['7', '1', '4', '6', '2', '5']
Invalid: ['3']
```

Requirements
------------
TODO

```
brew install cmake tesseract
