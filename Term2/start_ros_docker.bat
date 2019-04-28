REM create a new Container with File%
REM docker run -it -p 4567:4567 -v d:/EKF:/EKF udacity/controls_kit:latest
REM start an already container 
REM docker run --rm -it -p 4567:4567 -v D:\carND\Term2:/Term2 ros:latest
REM docker run -p 4567:4567 -v D:\carND\Term2:/Term2 --rm -it capstone
docker run -p 4567:4567 -v D:\carND\Term2\ros_log:/root/.ros/ -v D:\carND\Term2\:/Term2 -it capstone
