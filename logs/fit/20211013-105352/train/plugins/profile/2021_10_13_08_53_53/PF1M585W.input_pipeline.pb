$	?Ǡ????????@
h"lxz??!??ڊ?%+@$	?k??4?@eΡ??@ٍC\?`@!?82c?:@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ڊ?%+@M??St?@A??6 @Y+??N@"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????H?? c?ZB>??A????????Y??	h"l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ӽ??????T?????AX9??v???YjM??S??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?MbX9??5?8EGr??Aa2U0*???Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ZӼ???jM??S??A????߾??YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&TR'?????F%u???A?|a2U??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`???~??k	???ANё\?C??YY?8??m??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???镲???\?C????As??A???Ysh??|???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e?c??#??~j???A??????Y?Zd;??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??Q????M?O???A?+e?X??Y????镢?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
j?t???EGr????A???Q???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h ????y???A??h o???Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG????Q???A???߾??YP?s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&      ???'????A??C?l???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4??ꕲq???A?????M??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?w??#???؁sF????A???Q???YK?46??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"l????? ?r??A?46<??Y????镢?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?^)????`TR'???AF????x??Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\m?????????A&䃞ͪ??Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O????D???J??A?9#J{???Y???(\???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
h"lxz??	??g????A]?Fx??Y?ܵ?|У?*	????Ȳ@2F
Iterator::Model?[ A1@!??n?_U@)?????@1?6M:"T@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU0*????!,
j?  @)OjM???1s???>?@:Preprocessing2U
Iterator::Model::ParallelMapV2?I+???!#?3ĮH@)?I+???1#?3ĮH@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipR???Q??!?X?|?/@)S?!?uq??1+Dm?%F	@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???????!?l/%8? @)???????1?l/%8? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateI.?!????!bL?C?@)	?c?Z??1?O9????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapK?46??!0??U@)?HP???1?ÇU@=??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*\ A?c̝?!G??^??)\ A?c̝?1G??^??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t48.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?B3yk*@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?' ????????R??	??g????!M??St?@	!       "	!       *	!       2$	???	????")B%????]?Fx??!??6 @:	!       B	!       J$	???~:5??VvT??M????H?}??!+??N@R	!       Z$	???~:5??VvT??M????H?}??!+??N@JCPU_ONLYY?B3yk*@b 