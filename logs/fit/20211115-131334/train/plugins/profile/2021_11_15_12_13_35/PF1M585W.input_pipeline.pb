$	??~:5???T????????D????!????߾??$	?r^2?|@?o?Q?@^??!E@! 9???{1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??????????߾??A4??@????Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0?????ׁs??A??:M??Y?"??~j??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[???lV}???A????<,??Y:??H???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'?W?????(???Aq?-???Y??4?8E??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???9#J???Q???A?.n????YEGr????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}??????H.??A-??????Y??	h"l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V-?????????ApΈ?????Y?-????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?^)?????St$???AP??n???Y_?Qګ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l????????M??A?X????YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?.n?????ŏ1w??AgDio????Y?j+??ݣ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
???<,????W?2ı??A~??k	???Y??|?5^??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???9#J????h o???AO@a????Y?-?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}??ꕲq???A??<,Ԛ??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o?ŏ1???:pΈ??Ah??s???Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ׁsF????HP???A?鷯??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????߾??ꕲq???A+?????Y㥛? ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??z6?>W??Aa??+e??Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????߾??/?$???A&S????YTt$?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8EGr??????3???A?2ı.n??Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??\m????&䃞ͪ??A??6?[??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??D????s??A϶?A?߾?3??Y?o_???*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?d?`TR??!B???D@)?-???1??1? ?!+?C@:Preprocessing2F
Iterator::Model?????K??!??VE??@@);?O??n??1T@o(g?5@:Preprocessing2U
Iterator::Model::ParallelMapV2R???Q??!C?|??'@)R???Q??1C?|??'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$???????!?T]"?P@)?C?l????1婲w[?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicey?&1???!?m?@)y?&1???1?m?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate\???(\??!sn?<??&@)?5^?I??1Ho?^j?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|??Pk???!*??j??.@)??ܵ?|??1m=\?(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*/n????!?????@)/n????1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??a@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	QM??????^????s??A϶?!/?$???	!       "	!       *	!       2$	N???P?????'e)???߾?3??!q?-???:	!       B	!       J$	?2z[????S?~͍?????o??!??4?8E??R	!       Z$	?2z[????S?~͍?????o??!??4?8E??JCPU_ONLYY??a@b 