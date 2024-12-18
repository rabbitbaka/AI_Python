package com.example.news.controller;

import com.example.news.domain.News;
import com.example.news.dto.NewsDto;
import com.example.news.mapper.NewsMapper;
import com.example.news.repository.NewsRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

@Controller
@RequiredArgsConstructor // 초기화할때 필요한 필수값을 만들어줌 : final
@RequestMapping("/news")
public class NewsController {
    private final NewsMapper mapper; // 의존성 투입
    private final NewsRepository newsRepository;

    @GetMapping("/new")
    public String newArticleForm() {
        return "news/new";
    }

    @PostMapping("/create")
    public String createNews(NewsDto.Post post) {
        News news = mapper.newsPostDtoToNews(post); // mapper 가져오기
        newsRepository.save(news);
        return "redirect:/news/" + news.getNewsId();
    }

    // 단일 정보
    @GetMapping("/{newsId}")
    public String getNews(@PathVariable Long newsId, Model model){
        News news = newsRepository.findById(newsId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "해당 게시글이 존재하지 않습니다."));
        model.addAttribute("news", news);
        return "news/detail";
    }

    // 여러 정보 가져오기
    @GetMapping("/list")
    public String getNewsList(Model model,
                              @RequestParam(name = "page", defaultValue = "0") int page) {
        Pageable pageable = PageRequest.of(page, 5);
        Page<News> newsPage = newsRepository.findAll(pageable);
        model.addAttribute("newsPage", newsPage);

        model.addAttribute("prev", pageable.previousOrFirst().getPageNumber());
        model.addAttribute("next", pageable.next().getPageNumber());
        model.addAttribute("hasNext", newsPage.hasNext());
        model.addAttribute("hasPrev", newsPage.hasPrevious());
        return "news/list";
    }

    @GetMapping("/{newsId}/delete")
    public String deleteNews(@PathVariable("newsId") Long newsId){
        newsRepository.deleteById(newsId);
        return "redirect:/news/list";
    }

    @GetMapping("/{newsId}/edit")
    public String editNewsForm(@PathVariable("newsId") Long newsId, Model model) {
        News news = newsRepository.findById(newsId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "해당 뉴스가 존재하지 않습니다."));
        model.addAttribute("news", news);
        return "news/edit";
    }

    @PostMapping("/{newsId}/update")
    public String editNews(@PathVariable("newsId") Long newsId, NewsDto.Patch patch) {
        News news = newsRepository.findById(newsId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "해당 뉴스가 존재하지 않습니다."));
        mapper.PatchDtoToNews(patch, news);
        newsRepository.save(news);
        return "redirect:/news/" + news.getNewsId();
    }
}
