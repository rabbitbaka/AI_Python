package com.example.news.advice;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.server.ResponseStatusException;

@ControllerAdvice
public class GlobalExceptionHandler {
    // 응답 예외
    @ExceptionHandler(value = {ResponseStatusException.class})
    public String handleResponseStatusException(ResponseStatusException ex, Model model, HttpServletRequest request) {
        String currentPath = request.getRequestURI();
        model.addAttribute("path", currentPath);
        model.addAttribute("status", ex.getStatusCode().value());
        model.addAttribute("message", ex.getReason());
        return "error/error";
    }

    // 다른 예외
    @ExceptionHandler(value = {Exception.class})
    @ResponseStatus
    public String handleAllOtherExceptions(Exception ex, Model model, HttpServletRequest request){
        String currentPath= request.getRequestURI();
        model.addAttribute("path", currentPath);
        model.addAttribute("status", 500);
        model.addAttribute("message",ex.getMessage()+" 서버 오류가 발생했습니다.");
        return "error/error";
    }
}