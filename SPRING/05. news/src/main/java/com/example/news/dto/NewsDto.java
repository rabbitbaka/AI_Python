package com.example.news.dto;

import lombok.*;
public class NewsDto {
    @Getter
    @Setter
    @ToString // 주소가 아닌 NewsDto.Post(title=Title, content=속보입니다.)로 보이게 하는 것
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Post{
        private String title;
        private String content;
    }

    @Getter
    @Setter
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Patch {
        private String title;
        private String content;
    }
}
