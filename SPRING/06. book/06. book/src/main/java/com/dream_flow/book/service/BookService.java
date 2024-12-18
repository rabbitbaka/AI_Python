package com.dream_flow.book.service;

import com.dream_flow.book.domain.Book;
import com.dream_flow.book.repository.BookRepository;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
@Transactional
public class BookService {

    private final BookRepository bookRepository;

    // 책을 등록
    public Book inserBook(Book book){
//        Book b = bookRepository.save(book);
//        return b;
        return bookRepository.save(book); // 비영속 book을 영속으로
    }
    // 책을 업데이트(PUT)
    public Book updateBook(Long id, Book book){
        Book b = getBook(id);
        b.setTitle(book.getTitle());
        b.setSubTitle(book.getSubTitle());
        b.setAuthor(book.getAuthor());
        b.setPublisher(book.getPublisher());
        b.setStatus(book.getStatus());
        return bookRepository.save(b);
    }

    private Book getBook(Long id) {
        return bookRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 책입니다."));
    }

    // 책을 업데이트(PATCH)
    public Book updateBook(Long id, Book.Status status){
        Book b = getBook(id);
        b.setStatus(status);
        return bookRepository.save(b);
    }

    // 책을 삭제
    public void deleteBook(Long id) {
        // 비즈니스 로직 (대출 중인 책 확인)
        Book b = getBook(id);
        if(b.getStatus() == Book.Status.BORROWED){
            throw new IllegalArgumentException("대출 중인 책은 삭제할 수 없습니다.");
        }
        // 지우는 코드
        bookRepository.delete(b);
    }

    // 책을 조회(단건)
    public Book findBook(Long id){
        return getBook(id);
    }

    // 책을 조회(다건)
    public List<Book> findBooks() {
        return bookRepository.findAll();
    }
}