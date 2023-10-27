# Captcha-Solver
Project 2 for Scalable Computing - CS7NS1

## Raspberry Pi 2 Group Members

|    Student Name    | Student ID |                Course                |      GitHub Username       |
|:------------------:|:----------:|:------------------------------------:|:--------------------------:|
|   Prathamesh Sai   |  19314123  | Integrated Computer Science (M.C.S.) |    [saisankp][saisankp]    |
|     Hamzah Khan    |  23335427  | Computer Science - Data Science      |    [hamzah7k][hakhan]      |

This project requires you to have Python 2.7.18.

# To run everything on the Raspberry Pi
```
chmod +x run.sh
./run.sh
```

# To setup everything on the Raspberry Pi
Everything should already be setup (except Tensorflow which you must install manually as putting it in the /env/ on GitHub causes issues because of its size)
```
chmod +x setup.sh
./setup.sh
```

# To run everything on a Windows machine
```
chmod +x windowsRun.sh
./windowsRun.sh
```

# To setup everything on a Windows machine
Everything should already be setup (except Tensorflow which you must install manually as putting it in the /env/ on GitHub causes issues because of its size)
```
chmod +x windowsSetup.sh
./windowsSetup.sh
```

To solve captchas, we use different machines to minimise the time spent running our code. From our experiments we did:
1. Data collection from the server on the Raspberry Pi
2. Training captcha generation on a Windows machine
3. Processed our generated captchas on a Windows machine
4. Trained our CNN model on a Windows machine
5. Classified the images on the Raspberry Pi


[saisankp]: https://github.com/saisankp
[hakhan]: https://github.com/hamzah7k
 
