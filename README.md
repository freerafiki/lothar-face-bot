# lothar-face-bot
ì
A telegram bot for recognizing lothar faces

# Get it to run

## As python script ()
Simply run
```
python3 lothar-face-bot.py
```
And check that everything works as expected

## As service

##### Edit the .service file
Edit the `lothar-face-bot.service` file. Most likely you need to change
- `WorkingDirectory` (line 13): find it using the `pwd` command on linux
- `ExecStart` (line 14): find python using `which python` and the path to where you saved the python file, the full path to `/lothar-face-bot/lothar-face-bott.py`
- `ExecStop` (line 15): find pkill using `which pkill` and copy the full path (sometimes pkill alone is not working)
- `StandardOutput` and `StandardError` (line 21 and 22): change the full path to something existing on your server

##### Launch the service
First, reload the daemon system (it may require `sudo`, but not sure)
```
systemctl daemon-reload
```
and then start the service (also here same with `sudo`)
```
systemctl start lothar-face-bot.service
```
If it does not say anything, it worked! Otherwise check errors.
Use
```
systemctl status lothar-face-bot.service
```
to check the status.
