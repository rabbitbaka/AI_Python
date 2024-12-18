package com.example.jpa;

import com.example.jpa.domain.News;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class JpaApplication {

    public static void main(String[] args) {

        SpringApplication.run(JpaApplication.class, args);
        News n1 = new News();
    }

}
