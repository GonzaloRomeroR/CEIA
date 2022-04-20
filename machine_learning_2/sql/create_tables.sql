USE airportDB;

DROP TABLE test;
DROP TABLE train;

CREATE TABLE IF NOT EXISTS train(
	key_val int,
    id int, 
    Gender varchar(255),
    `Customer Type` varchar(255),
    Age int,
    `Type of Travel` varchar(255),
    Class varchar(255),
    `Flight Distance` int,
    `Inflight wifi service` int,
    `Departure/Arrival time convenient` int,
    `Ease of Online booking` int,
    `Gate location` int,
    `Food and drink` int,
    `Online boarding` int,
    `Seat comfort` int,
    `Inflight entertainment` int,
    `On-board service` int,
    `Leg room service` int,
    `Baggage handling` int,
    `Checkin service` int,
    `Inflight service` int,
    Cleanliness int,
    `Departure Delay in Minutes` int,
    `Arrival Delay in Minutes` int,
    satisfaction varchar(255)
);


CREATE TABLE IF NOT EXISTS test(
	key_val int,
    id int, 
    Gender varchar(255),
    `Customer Type` varchar(255),
    Age int,
    `Type of Travel` varchar(255),
    Class varchar(255),
    `Flight Distance` int,
    `Inflight wifi service` int,
    `Departure/Arrival time convenient` int,
    `Ease of Online booking` int,
    `Gate location` int,
    `Food and drink` int,
    `Online boarding` int,
    `Seat comfort` int,
    `Inflight entertainment` int,
    `On-board service` int,
    `Leg room service` int,
    `Baggage handling` int,
    `Checkin service` int,
    `Inflight service` int,
    Cleanliness int,
    `Departure Delay in Minutes` int,
    `Arrival Delay in Minutes` int,
    satisfaction varchar(255)
);


