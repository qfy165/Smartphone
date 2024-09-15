Welcome to the Smartphone Recommender System.
To run the code:
1. Upload requirements.txt, smartphone.csv, and smartphone.py to a new GitHub repository
2. Open streamlit.io and login with your github account
3. Press create app and select I have an app
4. Choose the new repository you have just created
5. Choose main as the branch
6. Choose smartphone.py as the main file path
7. Press Deploy!

Using the recommender system:
1. On the sidebar, you can choose between recommender system 1 and recommender system 2
2. For **recommender system 1**, use the dropdown menu to choose the smartphone of your choice or erase the current option and type it yourself
3. You can choose the number of recommendations you want to see by using the slider
4. The results will be updated automatically
5. For **recommender system 2**, you can choose the smartphone brand of your choice and the processor brands that the chosen smartphone brand offers
6. Alternatively, you can choose only the smartphone brand you want and leave the processor brand as any processor brand and vice versa
7. Next, you can move on to adjusting the price, battery, RAM, memory, screen size, as well as front and rear camera resolution
8. Press the submit button when you're done and the results will display all the smartphones that fit the criterias you just set

Note:
Currently, there are two variations of the code with only a minor difference in code. 
We discovered that version 1 doesn't work on every device due to this difference in code. 
If the version 1 of the code has errors and is unable to run, using version 2 will most likely solve the issue.

For version 2, just replace the code in line 150 & 151 with this:
'
if _name_ == "_main_":
    main()
'
