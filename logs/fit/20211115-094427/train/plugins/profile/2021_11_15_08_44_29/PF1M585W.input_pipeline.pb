$	![?2ހ?????ľ??-???????!d;?O????$	????x6@o?蟨q@UIa??" @!?????2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$-???????333333??A??H.?!??Y3ı.n???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???x?&???c]?F??Av??????YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y???H?}8g??A^K?=???Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&A?c?]K???d?`TR??A??ʡE???Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?T???N??	?c?Z??A??S㥛??Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&tF??_??"??u????A?J?4??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&R???Q?? ?o_???AV-???Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb??????z??AV-?????Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4???z6?>??A?A?f???Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?h o???E???JY??A??<,Ԛ??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
????????M?O???AȘ?????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???߾???j+?????A????߾??Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?J???????o??A?$??C??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??ݓ?????????A?&?W??Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?T???N??C?i?q???Aq???h??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}??????g??s??A?.n????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?ZB??lxz?,C??A????S??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*??D?????? ?r??AR???Q??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF???2ı.n??A?Zd;??Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d;?O????&S????A??h o???Y*??Dذ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8EGr?????a??4??A+??	h??Y???????*	fffffʈ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??e??a??!?aHAMA@)K?46??1??????@:Preprocessing2F
Iterator::ModelV????_??!"_,ueB@)#J{?/L??1?1W?z?5@:Preprocessing2U
Iterator::Model::ParallelMapV2鷯???! ?v,@)鷯???1 ?v,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W?2??!ޠӊ??O@)??Q???1A?????$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	?^)˰?!????ۉ @)	?^)˰?1????ۉ @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?u?????!?U!?1-@)ݵ?|г??1gt]p?O@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapd]?Fx??!l?@?N,3@)????镢?1?j?M@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*46<???!?x??]t@)46<???1?x??]t@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9.kR??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	w{??)???zx?????333333??!?c]?F??	!       "	!       *	!       2$	???hC???)??Km??+??	h??!??h o???:	!       B	!       J$	4Q??????y?????y?&1???!3ı.n???R	!       Z$	4Q??????y?????y?&1???!3ı.n???JCPU_ONLYY.kR??@b 