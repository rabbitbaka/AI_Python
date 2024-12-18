package com.example.news.domain;

import com.example.news.dto.NewsDto;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class News {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long newsId;

    @Column(name="NEWS_TITLE", nullable = false)
    private String title;

    @Column(nullable = false)
    private String content;

//    public static News toEntity(NewsDto.Post post){
//        return new News(null, post.getTitle(), post.getContent());
//    }
}
