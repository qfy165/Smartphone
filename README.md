Currently, there are two variations of the code with only a minor difference in code. 
We discovered that version 1 doesn't work on every device due to this difference in code. 
If the version 1 of the code has errors and is unable to run, using version 2 will most likely solve the issue.

For version 2, just replace the code in line 150 & 151 with this:
if _name_ == "_main_":
    main()
