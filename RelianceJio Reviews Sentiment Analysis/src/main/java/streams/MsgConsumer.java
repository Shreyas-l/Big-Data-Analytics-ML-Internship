package streams;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

public class MsgConsumer {


    public static KafkaConsumer consumer;

    public static void main(String[] args) throws IOException {
        configureConsumer(args);

        String topic = "/user/mapr/stream:reviews";
        if (args.length == 1) {
            topic = args[0];
        }

        List<String> topics = new ArrayList<>();
        topics.add(topic);
      
        consumer.subscribe(topics);

        long pollTimeOut = 1000;
        long waitTime = 30 * 1000;  // loop for while loop 30 seconds
        long numberOfMsgsReceived = 0;
        while (waitTime > 0) {
         
            ConsumerRecords<String, String> msg = consumer.poll(pollTimeOut);
            if (msg.count() == 0) {
            } else {
               
                numberOfMsgsReceived += msg.count();

                Iterator<ConsumerRecord<String, String>> iter = msg.iterator();
                while (iter.hasNext()) {
                    ConsumerRecord<String, String> record = iter.next();
                    System.out.println(record.value());

                }
            }
            waitTime = waitTime - 1000; // decrease time for loop
        }
        consumer.close();

    }


    public static void configureConsumer(String[] args) {
        Properties props = new Properties();
        props.put("auto.offset.reset", "earliest");
        props.put("key.deserializer",
                "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("value.deserializer",
                "org.apache.kafka.common.serialization.StringDeserializer");

        consumer = new KafkaConsumer<>(props);
    }

}
