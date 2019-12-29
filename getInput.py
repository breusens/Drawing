def getInput():
    x=input('y/n')
    if x=='n':
        y=0
    else:
        if x=='y':
            y=1
        else:
            y=getInput()
    return y


